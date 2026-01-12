import pickle
import numpy as np
import sys

# 定义一个极简的容器类
class SimpleShim:
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getattr__(self, name):
        return None

# 通过继承重写 find_class
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 拦截所有 chumpy 相关的类，统一指向 SimpleShim
        if 'chumpy' in module:
            return SimpleShim
        # 对于 numpy 和其他内置类型，尝试正常加载
        try:
            return super().find_class(module, name)
        except Exception:
            return SimpleShim

def inspect_smpl_pkl(pkl_path):
    # 彻底伪装模块，防止 pickle 内部触发 import
    m = SimpleShim()
    sys.modules['chumpy'] = m
    sys.modules['chumpy.ch'] = m

    with open(pkl_path, 'rb') as f:
        # 使用自定义的解包器
        unpickler = SafeUnpickler(f, encoding='latin1')
        data = unpickler.load()

    print(f"\n{'Key':<15} | {'Type':<20} | {'Shape/Info'}")
    print("-" * 65)

    for k, v in data.items():
        if isinstance(v, SimpleShim):
            # 从 chumpy 对象的字典中提取名为 'v' 的数组
            val = v.__dict__.get('v', None)
            shape = val.shape if hasattr(val, 'shape') else "No 'v' data"
            print(f"{k:<15} | Chumpy (Shim)      | {shape}")
        elif isinstance(v, np.ndarray):
            print(f"{k:<15} | Numpy Array       | {v.shape}")
        elif hasattr(v, 'shape') and hasattr(v, 'todense'):
            print(f"{k:<15} | Sparse Matrix    | {v.shape}")
        elif isinstance(v, (list, dict)):
            print(f"{k:<15} | {type(v).__name__:<18} | len={len(v)}")
        else:
            print(f"{k:<15} | {type(v).__name__:<18} | {v}")

if __name__ == "__main__":
    target = '../../Data/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    try:
        inspect_smpl_pkl(target)
    except Exception as e:
        import traceback
        traceback.print_exc()

def check_pkl():
    import numpy as np

    data = np.load("../../Data/smpl/basicModel_f_lbs_10_207_9_v1.0.0.npz")

    for k in data.files:
        arr = data[k]
        print(f"{k:15s} shape={arr.shape}, dtype={arr.dtype}")