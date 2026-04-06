# Project DQN Drone
## Cấu trúc
### File python thuần
Tải hết requirements xuống, lỗi tự fix, chỉ là lỗi dependencies thôi
Chạy: python train.py để nó chạy 200 episode sẽ có được file model.pth
Chạy: python export_onnx.py sẽ lấy model.pth ra được file mạng neuron model.onnx
### File model.onnx
Đây là cái mạng neuron sau khi train, đã có trọng số, giờ đem vô airsim hoặc bất kì cái phần mềm, môi trường nào demo thì tự xử tiếp.
## Cốt lõi ý tưởng
Cái Project này chủ yếu để có cái file onnx để chạy demo, hiện tại drone đang học cách để bay từ điểm A tới điểm B, tọa độ A,B đang hardcode trong file env.py nên vô chỉnh thoải mái, demo cũng sẽ demo ở tọa độ A,B theo code nha:  
    ```python
        self.A = np.array([0, 0, 1], dtype=np.float32)
        self.B = np.array([5, 5, 3], dtype=np.float32)
    ```
