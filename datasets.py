import json
import os, glob
import os.path as osp
import torch
from torchvision import datasets
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import tarfile
import io
from typing import Union
from pycocotools.coco import COCO
from torch.utils.data._utils.collate import default_collate
from typing import List, Dict, Tuple, Optional
import re
from torch.utils.data import ConcatDataset as _ConcatDataset
from tqdm import tqdm
import torch
import os.path as osp

import time


import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import torch
from torch.utils.data import Dataset
import random
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _get_file_len_worker(fpath):
    """并行子进程执行函数：返回 (fpath, 样本数, 错误信息)"""
    try:
        batch = torch.load(fpath, map_location="cpu")
        n = len(batch)
        del batch
        return fpath, n, None
    except Exception as e:
        return fpath, 0, str(e)


class CachedFeatureDataset_Packed(Dataset):
    """
    文件级采样 + 自动跨文件补齐 + 懒加载索引 + meta缓存 + 并行扫描版本
    """

    def __init__(self, cache_dirs, ar_target_layer: int = 10,
                 target_batch_size: int = 4, cache_meta: bool = True):
        if isinstance(cache_dirs, str):
            cache_dirs = [cache_dirs]
        self.cache_dirs = cache_dirs
        self.ar_target_layer = ar_target_layer
        self.target_batch_size = target_batch_size
        self.cache_meta = cache_meta

        # === 收集所有 .pt 文件 ===
        self.files = []
        for d in cache_dirs:
            if not osp.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    self.files.append(osp.join(d, f))
        self.files.sort()

        print(f"[INFO] Found {len(self.files)} packed cache files.")
        self.file_lengths_cache = {}

        # === meta.json 缓存路径 ===
        meta_path = osp.join(cache_dirs[0], "meta.json")

        # === 读取已有 meta.json ===
        if cache_meta and osp.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                valid_files = set(self.files)
                for fp, n in meta.items():
                    if fp in valid_files:
                        self.file_lengths_cache[fp] = n
                print(f"[INFO] Loaded cached meta.json ({len(self.file_lengths_cache)} entries).")
            except Exception as e:
                print(f"[WARN] Failed to read meta.json: {e}")

        # === 若 meta 不存在或不完整，触发并行扫描 ===
        missing_files = [f for f in self.files if f not in self.file_lengths_cache]
        if len(missing_files) > 0:
            print(f"[INFO] Parallel scanning {len(missing_files)} files to build meta.json ...")

            n_workers = min(max(cpu_count() // 2, 4), len(missing_files))
            results = []
            with Pool(processes=n_workers) as pool:
                for fpath, n, err in tqdm(
                    pool.imap_unordered(_get_file_len_worker, missing_files),
                    total=len(missing_files),
                    desc="[Indexing cache]",
                    ncols=100
                ):
                    if err is not None:
                        print(f"[WARN] {osp.basename(fpath)}: {err}")
                    self.file_lengths_cache[fpath] = n
                    results.append((fpath, n))

            if cache_meta:
                try:
                    with open(meta_path, "w") as f:
                        json.dump(self.file_lengths_cache, f, indent=2)
                    print(f"[INFO] Saved updated meta.json to {meta_path}")
                except Exception as e:
                    print(f"[WARN] Failed to write meta.json: {e}")

        self.total_samples = sum(self.file_lengths_cache.values())
        print(f"[INFO] Total {self.total_samples} samples across {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def _load_file(self, idx):
        fpath = self.files[idx]
        try:
            batch = torch.load(fpath, map_location="cpu")
            self.file_lengths_cache[fpath] = len(batch)
            return batch
        except Exception as e:
            print(f"[ERROR] Failed to load {fpath}: {e}")
            new_idx = (idx + 1) % len(self.files)
            return self._load_file(new_idx)

    def __getitem__(self, idx):
        total_samples = []
        file_idx = idx

        while len(total_samples) < self.target_batch_size:
            batch = self._load_file(file_idx)
            n = len(batch)
            remaining = self.target_batch_size - len(total_samples)

            if n <= remaining:
                total_samples.extend(batch)
            else:
                total_samples.extend(random.sample(batch, k=remaining))

            file_idx = (file_idx + 1) % len(self.files)

        total_samples = total_samples[:self.target_batch_size]

        out_list = []
        for data in total_samples:
            out = {}
            for k, v in data.items():
                if torch.is_tensor(v):
                    v = v.detach().requires_grad_(False)
                    if k == "txt_hidden_states":
                        if v.dim() == 4:
                            v = v[:, self.ar_target_layer, :, :]
                        elif v.dim() == 3:
                            v = v[self.ar_target_layer]
                    out[k] = v
                else:
                    out[k] = v
            out_list.append(out)

        return out_list


# === collate_fn 展平 ===
def collate_fn_packed(batch):
    flat = [sample for sublist in batch for sample in sublist]
    out = {}
    keys = flat[0].keys()
    for k in keys:
        vals = [b[k] for b in flat]
        if torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals
    return out



class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dirs, ar_target_layer: int = 10, shard_mode: bool = False):
        """
        参数:
            cache_dirs: 单个目录路径，或目录路径列表
            ar_target_layer: qwen 目标层数
        """
        if isinstance(cache_dirs, str):
            cache_dirs = [cache_dirs]

        self.cache_dirs = cache_dirs
        self.ar_target_layer = ar_target_layer

        # 收集所有 .pt 文件
        self.files = []
        for d in cache_dirs:
            if not osp.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    self.files.append(osp.join(d, f))
        self.files.sort()



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = torch.load(path, map_location="cpu")  # 单个 dict
        except Exception:
            raise FileNotFoundError(f".pt File broken! ({path})")

        out = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                v = v.detach().requires_grad_(False)
                if k == "txt_hidden_states":
                    if v.dim() == 4:
                        v = v[:, self.ar_target_layer, :, :]  # (B, N, D)
                    elif v.dim() == 3:
                        v = v[self.ar_target_layer]           # (N, D)
                out[k] = v
            else:
                out[k] = v
        return out



class COCODataset(Dataset):
    def __init__(self, root, train, annFile, transform=None, instances_annFile=None):
        self.root = root
        self.train = train
        self.dataset = datasets.CocoCaptions(root=root, annFile=annFile, transform=transform)
        self.inst_coco = COCO(instances_annFile) if instances_annFile is not None else None

    def __getitem__(self, index):
        img, captions = self.dataset[index]
        # 取该样本对应的 image_id
        image_id = self.dataset.ids[index]

        # 采一条 caption（与原逻辑一致）
        if self.train:
            cindex = torch.randint(0, len(captions), (1,)).item()
        else:
            cindex = 0
        caption = captions[cindex]

        # 取该图片的类别（可能多类）
        cat_ids, cat_names = [], []
        if self.inst_coco is not None:
            ann_ids = self.inst_coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            anns = self.inst_coco.loadAnns(ann_ids)
            cat_ids = sorted({a["category_id"] for a in anns})
            if cat_ids:
                cats = self.inst_coco.loadCats(cat_ids)
                cat_names = [c["name"] for c in cats]

        # 新的返回：img, caption, cat_ids, cat_names
        return img, caption

    def __len__(self):
        return len(self.dataset)

    

class Blip3oDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.data_pairs = []
        self.tar_members = {}   # {tar_path: set(file_names)}
        self._tars = {}         # lazy-open per-process cache: {tar_path: TarFile}
        self._members_map = {}  # {tar_path: {name: TarInfo}}
        self._collect_data_pairs()

    def __getstate__(self):
        state = self.__dict__.copy()
        # tar 句柄不能被 pickle，丢弃，worker 中会按需重新打开
        state["_tars"] = {}
        state["_members_map"] = {}
        return state

    def _ensure_open(self, tar_path: str):
        tf = self._tars.get(tar_path)
        if tf is None:
            # 用 "r:" 而不是流式模式，支持随机访问
            tf = tarfile.open(tar_path, mode="r:")
            self._tars[tar_path] = tf
            # 首次打开时做成员字典，后续 O(1) 查找
            members = tf.getmembers()
            self._members_map[tar_path] = {m.name: m for m in members}
        return self._tars[tar_path]

    def _collect_data_pairs(self) -> None:
        tar_files = [f for f in os.listdir(self.root_dir)
                     if f.endswith('.tar') and os.path.isfile(os.path.join(self.root_dir, f))]

        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        # 只做轻量级目录扫描（不打开 tar）
        for tar_file in tar_files:
            tar_path = os.path.join(self.root_dir, tar_file)
            with tarfile.open(tar_path, 'r:') as tf:
                names = tf.getnames()
            self.tar_members[tar_path] = set(names)
            for name in names:
                if name.lower().endswith(image_exts):
                    base = os.path.splitext(name)[0]
                    txt = f"{base}.txt"
                    if txt in self.tar_members[tar_path]:
                        self.data_pairs.append((tar_path, name, txt))

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int):
        tar_path, img_name, txt_name = self.data_pairs[idx]
        tf = self._ensure_open(tar_path)

        # 用已缓存的 TarInfo，避免线性搜索
        members = self._members_map[tar_path]
        img_info = members.get(img_name)
        if img_info is None:
            raise FileNotFoundError(f"{img_name} not in {tar_path}")
        txt_info = members.get(txt_name)
        if txt_info is None:
            raise FileNotFoundError(f"{txt_name} not in {tar_path}")

        img_file_obj = tf.extractfile(img_info)
        if img_file_obj is None:
            raise FileNotFoundError(f"{img_name} unreadable in {tar_path}")
        # 直接把文件对象交给 PIL，避免 BytesIO 复制
        image = Image.open(img_file_obj).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        txt_file_obj = tf.extractfile(txt_info)
        if txt_file_obj is None:
            raise FileNotFoundError(f"{txt_name} unreadable in {tar_path}")
        caption = txt_file_obj.read().decode("utf-8").strip()

        return image, caption


class EchoImage4oDataset(Dataset):
    """
    目录结构（root_dir = <datadir>/Echo-4o-Image）：
      root_dir/
        SubsetA/                 # 例如：Surrel-Fantasy-Image
          *.json                 # 仅一个，记录 instruction / output_image / type
          images/
            0-5000.tar.gz
            5000-10000.tar.gz
            ...
        SubsetB/                 # 例如：Instruction-Following-Image
          *.json
          images/
            ...
    JSON 可为 JSON Lines（每行一个对象）或 JSON 数组。
    每条记录示例：
      {"task_type":"t2i","instruction":"...","output_image":"/Echo-4o-Image/Surrel-Fantasy-Image/images/00000.jpg","type":"T1"}
    """
    def __init__(
        self,
        root_dir: str,
        transform=None,
        subsets: Optional[List[str]] = None,      # 指定只用哪些子集；默认全部子目录
        keep_types: Optional[List[str]] = None,    # 例如：只用 ["T1","easy"]
        drop_missing: bool = True,                 # 找不到图像条目时是否丢弃
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.keep_types = set(keep_types) if keep_types else None
        self.drop_missing = drop_missing

        # 索引：每个子集分别扫 images/*.tar.gz 的成员名，建立 {tar_path: set(names)}
        self._tar_name_sets: Dict[str, set] = {}
        # 运行期懒打开：{tar_path: TarFile}
        self._tars: Dict[str, tarfile.TarFile] = {}
        # 运行期成员快速索引：{tar_path: {name: TarInfo}}
        self._members_map: Dict[str, Dict[str, tarfile.TarInfo]] = {}

        # 条目：[(tar_path, member_name, instruction)]
        self.items: List[Tuple[str, str, str]] = []

        # 选子集
        all_subdirs = sorted([d for d in os.listdir(self.root_dir)
                              if os.path.isdir(os.path.join(self.root_dir, d))])
        use_subdirs = subsets if subsets else all_subdirs
        for sub in use_subdirs:
            sub_dir = os.path.join(self.root_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            # 1) 找 JSON（取第一个 *.json）
            json_files = sorted(glob.glob(os.path.join(sub_dir, "*.jsonl")))
            if not json_files:
                continue
            json_path = json_files[0]

            # 2) 扫描该子集的 tar.gz 并记录成员名集合
            images_dir = os.path.join(sub_dir, "images")
            tar_paths = sorted(glob.glob(os.path.join(images_dir, "*.tar.gz")))
            local_tar_name_sets = {}
            for tp in tar_paths:
                try:
                    with tarfile.open(tp, mode="r:gz") as tf:
                        names = set(tf.getnames())
                except tarfile.TarError:
                    names = set()
                self._tar_name_sets[tp] = names
                local_tar_name_sets[tp] = names

            # 3) 读取 JSON 记录
            records = self._read_json_records(json_path)

            for rec in records:
                # 过滤类型（可选）
                if self.keep_types and rec.get("type") not in self.keep_types:
                    continue

                instr = rec.get("instruction", "").strip()
                outp = rec.get("output_image", "").strip()
                if not outp or not instr:
                    if self.drop_missing:
                        continue
                    else:
                        instr = instr or ""
                        outp = outp or ""

                # 从 output_image 取到文件名，例如 "00000.jpg"
                base = os.path.basename(outp)
                if not base:
                    if self.drop_missing:
                        continue
                    else:
                        base = ""

                # 在本子集的 tar.gz 里查找该文件
                match = None
                member_key = None
                for tp, name_set in local_tar_name_sets.items():
                    # 可能存在不同打包路径：直接文件名或带 images/ 前缀
                    if base in name_set:
                        match = tp
                        member_key = base
                        break
                    prefixed = f"images/{base}"
                    if prefixed in name_set:
                        match = tp
                        member_key = prefixed
                        break
                if match is None:
                    # 找不到：按策略处理
                    if self.drop_missing:
                        continue
                    else:
                        # 允许缺失则占位（取第一个 tar，member_key 也占位，后续 __getitem__ 仍可能失败）
                        if tar_paths:
                            match = tar_paths[0]
                            member_key = base

                self.items.append((match, member_key, instr))

    def __getstate__(self):
        # DataLoader 多进程下，tar 句柄不可序列化，清空，worker 内按需重建
        state = self.__dict__.copy()
        state["_tars"] = {}
        state["_members_map"] = {}
        return state

    def _ensure_open(self, tar_path: str) -> tarfile.TarFile:
        tf = self._tars.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, mode="r:gz")
            self._tars[tar_path] = tf
            # 首次打开时建成员索引，后续 O(1) 查找
            members = tf.getmembers()
            self._members_map[tar_path] = {m.name: m for m in members}
        return tf

    @staticmethod
    def _read_json_records(json_path: str) -> List[dict]:
        # 同时兼容 JSON Lines 与 JSON 数组两种写法
        with open(json_path, "r", encoding="utf-8") as f:
            head = f.read(1)
            if not head:
                return []
            f.seek(0)
            if head == "[":
                data = json.load(f)
                return data if isinstance(data, list) else []
            else:
                records = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # 容错：有的文件可能没有换行分隔，这里尝试用 '}{' 拆分
                        # 仅当整文件在一行时启用
                        pass
                if not records:
                    # 退化处理：将整个文件读入再按 '}{' 粗分
                    f.seek(0)
                    blob = f.read().strip()
                    if blob.startswith("{") and blob.endswith("}"):
                        parts = blob.replace("}{", "}\n{").splitlines()
                        for p in parts:
                            try:
                                records.append(json.loads(p))
                            except Exception:
                                pass
                return records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        tar_path, member_name, instruction = self.items[idx]

        # 保险处理
        if (tar_path is None) or (not member_name):
            return None if self.drop_missing else (None, instruction)

        tf = self._ensure_open(tar_path)
        members = self._members_map.get(tar_path, {})
        m = members.get(member_name)
        if m is None:
            # 有些打包路径可能还带 './' 前缀
            alt = member_name.lstrip("./")
            m = members.get(alt)
            if m is None:
                return None if self.drop_missing else (None, instruction)

        fobj = tf.extractfile(m)
        if fobj is None:
            return None if self.drop_missing else (None, instruction)

        img = Image.open(fobj).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # 与其他数据集对齐：返回 (image, caption)
        return img, instruction
    

def get_target_dataset(name: str, datadir, train=False, transform=None):

    if name == 'coco':
        datapath = os.path.join(datadir, 'coco', 'train2017') if train else os.path.join(datadir, 'coco', 'val2017')
        cap_ann = os.path.join(datadir, 'coco', 'annotations',
                               'captions_train2017.json' if train else 'captions_val2017.json')
        inst_ann = os.path.join(datadir, 'coco', 'annotations',
                                'instances_train2017.json' if train else 'instances_val2017.json')
        dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                              transform=transform, instances_annFile=inst_ann)

    elif name == 'blip3o':
        if train:
            datapath = os.path.join(datadir, 'BLIP3o', 'BLIP3o-Pretrain-Long-Caption-subset-0') 
            dataset = Blip3oDataset(root_dir=datapath, transform=transform)
        else:
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)


    elif name == 'blip3o60k':
        if train:
            datapath = os.path.join(datadir, 'BLIP3o', 'BLIP3o-60k') 
            dataset = Blip3oDataset(root_dir=datapath, transform=transform)
        else:
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)
    elif name == 'Echo-4o-Image':
        if train:
            datapath = os.path.join(datadir, 'Echo-4o-Image')
            # 可按需筛子集或类型：subsets=['Surrel-Fantasy-Image','Instruction-Following-Image'], keep_types=['T1','easy']
            dataset = EchoImage4oDataset(root_dir=datapath, transform=transform, subsets=['Surrel-Fantasy-Image', 'Instruction-Following-Image'])
        else:
            # 评测沿用 COCO val（与 blip3o* 的做法一致）
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)
            
    elif name == 'Echo-4o-Image-Fantasy':
        if train:
            datapath = os.path.join(datadir, 'Echo-4o-Image')
            # 可按需筛子集或类型
            dataset = EchoImage4oDataset(root_dir=datapath, transform=transform, subsets=['Surrel-Fantasy-Image'])
        else:
            # 评测沿用 COCO val（与 blip3o* 的做法一致）
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)
            
    elif name == 'Echo-4o-Image-Instruction':
        if train:
            datapath = os.path.join(datadir, 'Echo-4o-Image')
            # 可按需筛子集或类型：
            dataset = EchoImage4oDataset(root_dir=datapath, transform=transform, subsets=['Instruction-Following-Image'])
        else:
            # 评测沿用 COCO val（与 blip3o* 的做法一致）
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)
            
    elif name.startswith("blip3o60k"):
        if train:
            datapath = os.path.join(datadir, 'BLIP3o', name) 
            dataset = Blip3oDataset(root_dir=datapath, transform=transform)
        else:
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)  

    elif name.startswith("QwenImage200k"):
        if train:
            datapath = os.path.join(datadir, 'QwenImage200k', name) 
            dataset = Blip3oDataset(root_dir=datapath, transform=transform)
        else:
            datapath = os.path.join(datadir, 'coco', 'val2017')
            cap_ann = os.path.join(datadir, 'coco', 'annotations', 'captions_val2017.json')
            inst_ann = os.path.join(datadir, 'coco', 'annotations', 'instances_val2017.json')
            dataset = COCODataset(root=datapath, train=train, annFile=cap_ann,
                                  transform=transform, instances_annFile=inst_ann)       

     
    else:
        raise ValueError(f"Dataset {name} not supported.")
    return dataset
