from setuptools import setup, find_packages

setup(
    name='sft_dpo_qlora',
    version='0.1.0',
    description='SFT-DPO-QLora Trainer Package',
    author='Sherma Thangam S',
    author_email='sshermathangam@gmail.com',
    url='https://github.com/Sherma-ThangamS/SFT-DPO-QLora',
    packages=find_packages(),
    install_requires=[
        'torch',
        'datasets',
        'peft',
        'transformers',
        'trl',
        'bitsandbytes', 
        'auto-gptq',
        'optimum',
        'einops'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='sft dpo qlora trainer',
    python_requires='>=3.9',
    entry_points={
    'console_scripts': [
        'sft_train=sft_dpo_qlora.trainerSFT:main',
        'dpo_train=sft_dpo_qlora.trainerDPO:main',
        ],
    },
)
