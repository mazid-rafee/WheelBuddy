# WheelBuddy website setup

This package is designed to overlay your existing `DriveSensAI` repository. It does not include your videos or app screenshot because those files are on your Mac.

## 1. Copy the files

Unzip the package. Copy its `website/index.html`, `website/styles.css`, `website/script.js`, and `website/favicon.svg` into:

`/Users/zaimazarnaz/Desktop/IOSApps/DriveSensAI/website/`

Keep your existing `website/media/` folder. Copy `.github/workflows/pages.yml` from the package to the matching path at the root of your Git repository. If you already have a Pages workflow, inspect it before replacing it.

The resulting tree should include:

```text
DriveSensAI/
├── .github/workflows/pages.yml
└── website/
    ├── index.html
    ├── styles.css
    ├── script.js
    ├── favicon.svg
    └── media/
        ├── app_landing_page.jpg
        └── live_test_videos/
            ├── Drive Drowsiness Safety.mp4
            ├── Driver Attention Safety.mp4
            ├── Front Car Crash Safety.mp4
            ├── Lane_Detection.mp4
            ├── Pedestrian Safety.mp4
            └── Road Safety Navigation.mp4
```

Filenames are case-sensitive on GitHub Pages. This site uses the exact media names visible in your screenshot. If any actual filename differs, edit its `src` in `website/index.html` (for the image) or the corresponding video card's `data-video` attribute. Spaces in the video URLs are encoded as `%20`.

## 2. Preview on your Mac

From the repository root:

```bash
python3 -m http.server 8000 --directory website
```

Open `http://localhost:8000/` and test the image, tabs, video filters, and all six videos. Stop the server with `Control-C`.

## 3. Publish with GitHub Pages

In the repository on GitHub, go to **Settings → Pages → Build and deployment** and choose **GitHub Actions** as the source. Commit and push the new files to `main`:

```bash
git add website/index.html website/styles.css website/script.js website/favicon.svg .github/workflows/pages.yml
git add website/media/app_landing_page.jpg website/media/live_test_videos
git commit -m "Add WheelBuddy project website"
git push origin main
```

If your videos are already tracked, `git add` is harmless. Check `git status` before committing, especially if your repository has other work in progress. If a video is too large for a normal Git push, use Git LFS or host a compressed clip elsewhere and update its `data-video` URL. GitHub Pages also has site size and bandwidth limits.

After the workflow succeeds under **Actions**, visit **Settings → Pages** for the live URL. If the repository is `mazid-rafee/WheelBuddy`, it should be `https://mazid-rafee.github.io/WheelBuddy/`.

## Edit the content

- `index.html`: text, links, media paths, and the six video cards.
- `styles.css`: appearance and responsive layout.
- `script.js`: mobile menu, tabs, filters, video player, and entrance effects.

The app screenshot appears in the hero. The route and driving graphics are illustrative; video clips are the real tests. The IIHS findings are linked to their original studies and explicitly described as research on other vehicle systems, not WheelBuddy effectiveness.
