import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('src')

def test_data_loading():

    
    try:
        from src.utils import load_and_prepare_data, collate_fn
        
        data = load_and_prepare_data()
        
        print("Data loading successful")
        print(f"  - Train dataset: {len(data['train_dataset'])} sequences")
        print(f"  - Val dataset: {len(data['val_dataset'])} sequences")
        print(f"  - Test dataset: {len(data['test_dataset'])} sequences")
        print(f"  - Features: {data['num_features']}")
        print(f"  - Classes: {data['num_classes']}")
        
        train_loader = DataLoader(data['train_dataset'], batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        for batch in train_loader:
            seqs, labels, mask = batch
            print(f"  - Batch shape: {seqs.shape}")
            print(f"  - Labels shape: {labels.shape}")
            print(f"  - Mask shape: {mask.shape}")
            break
        
        print("DataLoader test successful")
        return True, data
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return False, None

def test_models(data):

    
    try:
        from src.model import get_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {device}")
        
        batch_size = 4
        seq_len = 50
        input_dim = data['num_features']
        num_classes = data['num_classes']
        
        x = torch.randn(batch_size, seq_len, input_dim).to(device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
        mask[:, -10:] = True 
        
        print("Testing CNN-Transformer...")
        model = get_model('cnn_transformer', input_dim, num_classes, hidden_dim=64).to(device)
        
        with torch.no_grad():
            output = model(x, mask)
            print(f"  - Output shape: {output.shape}")
            print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print("CNN-Transformer test successful")
        
        print("\nTesting BiLSTM...")
        rnn_model = get_model('rnn', input_dim, num_classes, hidden_dim=64).to(device)
        
        with torch.no_grad():
            rnn_output = rnn_model(x, mask)
            print(f"  - Output shape: {rnn_output.shape}")
            print(f"  - Parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
            print("BiLSTM test successful")
        
        return True
        
    except Exception as e:
        print(f"Model testing failed: {e}")
        return False

def test_training(data):

    
    try:
        os.makedirs('../results', exist_ok=True)
        os.makedirs('../results/models', exist_ok=True)
        
        from src.train import train_model
        
        print("Running mini training (2 epochs)...")
        result = train_model(model_name='cnn_transformer', num_epochs=2, batch_size=8, learning_rate=1e-3)
        
        print(f"Training test successful")
        print(f"  - Final Val F1: {result['final_val_f1']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Training test failed: {e}")
        return False

def test_cuda_availability():

    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        device = torch.device('cuda')
        x = torch.randn(100, 100).to(device)
        y = x @ x.T
        print(f"CUDA computation test successful")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    else:
        print("CUDA not available, will use CPU")
    
    return True

def test_file_structure():

    
    required_files = [
        'src/utils.py',
        'src/model.py',
        'src/train.py',
        'dataset/train.csv',
        'dataset/test.csv'
    ]
    
    required_dirs = [
        'src/',
        'dataset/',
        'results/'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"Directory exists: {dir_path}")
        else:
            print(f"Directory missing: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"File exists: {file_path}")
        else:
            print(f"File missing: {file_path}")
            all_good = False
    
    return all_good

def test_imports():

    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        print("PyTorch imports successful")
        
        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        print("ML libraries imports successful")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("Visualization libraries imports successful")
        
        from src.utils import load_and_prepare_data, CMISequenceDataset, collate_fn
        from src.model import get_model, CNNTransformer
        print("Custom modules imports successful")
        
        return True
        
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def run_comprehensive_test():

    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("CUDA Setup", test_cuda_availability),
        ("Data Loading", test_data_loading),
    ]
    
    results = {}
    data = None
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            if test_name == "Data Loading":
                success, data = test_func()
                results[test_name] = success
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    if results.get("Data Loading", False) and data is not None:
        print(f"\nRunning Model Architecture test...")
        try:
            results["Model Architecture"] = test_models(data)
        except Exception as e:
            print(f"Model Architecture test crashed: {e}")
            results["Model Architecture"] = False
        
        print(f"\nRunning Training Pipeline test...")
        try:
            results["Training Pipeline"] = test_training(data)
        except Exception as e:
            print(f"Training Pipeline test crashed: {e}")
            results["Training Pipeline"] = False
    

    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<20}: {status}")
        if success:
            passed += 1
    
    print(f"\n Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(" All tests passed")
    else:
        print(" Some tests failed.")
    
    return results

def quick_test():
    
    print("Testing imports...")
    if not test_imports():
        return False
    
    print("\nTesting CUDA...")
    test_cuda_availability()
    
    print("\nTesting file structure...")
    if not test_file_structure():
        print("Some files missing")
        return False
    
    print("\nquick test completed successfully!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BFRB detection system')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    parser.add_argument('--full', action='store_true', help='Run comprehensive test')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.full:
        run_comprehensive_test()
    else:
        quick_test()
