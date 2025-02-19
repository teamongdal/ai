import numpy as np

def load_npy_file(file_path, sample_size=5):
    """
    .npy 파일을 로드하고 일부 데이터를 출력하는 함수
    
    Parameters:
        file_path (str): .npy 파일 경로
        sample_size (int): 출력할 샘플 데이터 개수
    """
    try:
        data = np.load(file_path, allow_pickle=True)  # allow_pickle=True 옵션 사용
        print(f"파일 로드 성공: {file_path}")
        print(f"데이터 개수: {len(data)}")
        
        # 데이터 일부 출력
        print("샘플 데이터:")
        for i, sample in enumerate(data[:sample_size]):
            print(f"샘플 {i+1}({len(sample)}): {sample}\n")
        
        return data
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        return None

# 사용 예시
file_path = "product_features_2048.npy"  # 실제 파일 경로로 변경
npy_data = load_npy_file(file_path)
