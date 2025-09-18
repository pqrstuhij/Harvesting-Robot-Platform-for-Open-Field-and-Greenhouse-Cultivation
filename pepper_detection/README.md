# Red-Pepper-Detection

### 고추 탐지

- 개발기간 : 24.11.14 ~ 24.12.20
- 최신버전 : v0.1

## 시작 가이드

```
$ python -m venv venv
$ pip install requirements.txt
```

## 프로젝트 설명

고추 수확 로봇 시스템을 위한 성숙도 판별 및 최적 절단부 검출

## 프로젝트 히스토리

- 1차 작업 (24.11.14 ~ 진행중)
  - v0.1
  - YOLOv11 모델을 사용하여 고추 탐지

## 사용 기술

`Python` `YOLOv11` `OpenCV` 

## 프로젝트 파일 구조

```
/
├─data
│  ├─pepper
│  └─stem
├─models
│  └─pretrained
├─src
└─test_images
```

> data
>
> - 학습 데이터셋 폴더
> - pepper : 고추 이미지 데이터셋
> - stem : 줄기 이미지 데이터셋
>
> models
>
> - 학습된 모델 폴더
> - pretrained : 사전 학습된 모델 폴더
> - PSM : Pepper Stem Model
> - RPM : Red Pepper Model
>
> src
>
> - 소스코드 폴더
>
> test_images
>
> - 테스트 이미지 폴더