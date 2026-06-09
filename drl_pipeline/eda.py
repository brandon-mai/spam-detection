import numpy as np

def analyze(file_path):
    print(f"\n--- Analyzing {file_path} ---")
    data = np.load(file_path)
    V = data['V']
    E = data['E']
    src = data['src']
    tgt = data['tgt']
    quota = data['quota']
    
    print(f"Total transitions: {len(src)}")
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    print(f"E shape: {E.shape}, dtype: {E.dtype}")
    print(f"src shape: {src.shape}, dtype: {src.dtype}")
    
    # Check for NaNs/Infs
    print(f"V has NaNs: {np.isnan(V).any()}, Infs: {np.isinf(V).any()}")
    print(f"E has NaNs: {np.isnan(E).any()}, Infs: {np.isinf(E).any()}")
    
    # Action distribution
    no_op_mask = (tgt == 50) | ((src == 0) & (tgt == 50))
    n_no_op = np.sum(no_op_mask)
    n_send = len(src) - n_no_op
    print(f"\nAction Distribution:")
    print(f"  NO-OPs: {n_no_op} ({n_no_op/len(src)*100:.2f}%)")
    print(f"  SENDs:  {n_send} ({n_send/len(src)*100:.2f}%)")
    
    # Value ranges
    print("\nFeature ranges:")
    print(f"  V min: {np.min(V):.4f}, max: {np.max(V):.4f}")
    print(f"  E min: {np.min(E):.4f}, max: {np.max(E):.4f}")
    
    # Non NO-OP stats
    if n_send > 0:
        valid_tgt = tgt[~no_op_mask]
        valid_src = src[~no_op_mask]
        print(f"  Target nodes range: {np.min(valid_tgt)} to {np.max(valid_tgt)}")
        print(f"  Source nodes range: {np.min(valid_src)} to {np.max(valid_src)}")
        
        # Quota distribution
        valid_quota = quota[~no_op_mask]
        unique_q, counts_q = np.unique(valid_quota, return_counts=True)
        print("  Quota Distribution (for SENDs):")
        for u, c in zip(unique_q, counts_q):
            print(f"    Quota {u}: {c} ({c/len(valid_quota)*100:.2f}%)")

analyze("drl_pipeline/expert_dataset_2p.npz")
analyze("drl_pipeline/expert_dataset_4p.npz")
