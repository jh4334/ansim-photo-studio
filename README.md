# 안심사진관

학교 사진에서 학생 얼굴 위치를 감지하고, 블러/모자이크/검은 박스로 가린 뒤 PNG로 다운로드하는 로컬 실행 웹앱입니다.

## 실행

```bash
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173/`로 접속합니다.

배포 빌드 확인:

```bash
npm run build
npm run preview
```

## 공유 배포

GitHub Pages로 배포하면 다른 사람은 설치 없이 링크로 실행할 수 있습니다.

1. GitHub 저장소의 `Settings > Pages`에서 `Build and deployment` 소스를 `GitHub Actions`로 설정합니다.
2. `main` 브랜치에 push하면 `.github/workflows/deploy-pages.yml`가 자동으로 빌드/배포합니다.
3. 배포 주소는 보통 `https://jh4334.github.io/ansim-photo-studio/` 형식입니다.

권장 브라우저는 Chrome 또는 Edge입니다. `PNG 폴더 저장`은 브라우저의 폴더 선택 기능을 사용하므로, 사용자가 직접 저장할 폴더를 선택할 수 있습니다.

## 개인정보 보호

- 사진은 서버에 업로드하지 않습니다.
- 모든 처리는 사용자의 PC 브라우저 안에서만 수행됩니다.
- DB, 로그인, 서버 저장 기능이 없습니다.
- 특정 인물이 누구인지 식별하지 않고 얼굴 위치만 감지합니다.
- 자동 감지는 일부 얼굴을 놓칠 수 있으므로 게시 전 반드시 사용자가 최종 확인해야 합니다.

## 포함된 모델

`public/models` 폴더에 MediaPipe wasm 파일과 BlazeFace 모델 파일을 포함해 localhost에서 오프라인으로 실행할 수 있게 구성했습니다.
