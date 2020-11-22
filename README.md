# first-order_PaddlePaddle

为进入测试模式，请修改run.py中第20行附近：

```python
TEST_MODE = True			# 请改为True以进入测试模式
if TEST_MODE:
    logging.warning('TEST MODE: run.py')
    fake_batch_size = 2		# 修改batchsize, aistudio的gpu上size为8时不报错，为9时报错
    ......
```

aistudio上运行训练：

```bash
cd ./src
pip install scikit-image scipy
# use gpu
python run.py --config ./config/test-256.yaml
# use cpu
python run.py --config ./config/test-256.yaml --cpu
```



BatchSize为8时无错误，为9时modules.model中的144行附近类型转换报错。若不使用类型转换则在loss.backward()报错。

```python
def warp_coordinates(self, coordinates):
	theta = self.theta.astype('float32')						# 此处报错
	theta = theta.unsqueeze(1)
	coordinates = coordinates.unsqueeze(-1)
	theta_part_a = theta[:, :, :, :2]
	theta_part_b = theta[:, :, :, 2:]
	transformed = paddle.fluid.layers.matmul(*broadcast_v1(theta_part_a, coordinates)) + theta_part_b
	transformed = transformed.squeeze(-1)
	if self.tps:
		control_points = self.control_points.astype('float32')	# 此处报错
		control_params = self.control_params.astype('float32')	# 此处报错
		distances = coordinates.reshape((coordinates.shape[0], -1, 1, 2)) - control_points.reshape((1, 1, -1, 2))
		distances = distances.abs().sum(-1)
		result = distances * distances
		result = result * paddle.log(distances + 1e-6)
		result = result * control_params
		result = result.sum(2).reshape((self.bs, coordinates.shape[1], 1))
		transformed = transformed + result
	return transformed
```