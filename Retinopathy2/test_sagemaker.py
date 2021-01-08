import sagemaker

role = sagemaker.get_execution_role()

from sagemaker.pytorch import PyTorchModel

model_data = 's3://dataset-retinopathy/deployments/model.tar.gz'

model = PyTorchModel(model_data=model_data,
                     role=role,
                     entry_point='inference.py',
                     framework_version='1.6.0',
                     py_version='py3',
                     source_dir='code')


