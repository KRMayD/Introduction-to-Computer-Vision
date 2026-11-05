# CIFAR-10 데이터 준비

[공식 CIFAR-10 배포 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)의 **CIFAR-10 python version**을 준비합니다. 아카이브 내부에 `cifar-10-batches-py/data_batch_1`부터 `data_batch_5`, `test_batch`가 있어야 합니다.

기본 입력 파일 이름은 `data/cifar-10-python.tar`입니다. 파일 이름을 유지하려면 `--data /path/to/cifar-10-python.tar.gz`처럼 실제 경로를 전달합니다. 로더는 tar/gzip 압축을 자동 인식하며 아카이브를 수정하거나 압축을 풀지 않습니다.

실험에 사용한 로컬 tar의 SHA-256은 다음과 같습니다.

```text
d9f61bfd30fed3a3cc4d10e1cdaeb906ffc4005bfa2c59c06b4aeeede24027f1
```

압축을 다시 하거나 gzip 버전을 사용하면 파일 해시는 달라질 수 있습니다. 새 기본 실험은 전달한 파일의 해시를 기록하고, 후속 투표 비교 실험은 해당 기본 실험의 해시와 대조합니다.

데이터셋과 추출된 배치는 Git 업로드 대상에서 제외됩니다. 결과 표는 [results/README.md](../results/README.md), 실행 명령은 [상위 README](../README.md#실험-재현)에 있습니다.
