# Putting this project on GitHub (step by step, for a first-time user)

You need: the project folder (unzipped `cashflow-forecasting/`), an email address, and your
friend's GitHub username (or email).

---

## 1. Create your GitHub account (≈3 minutes)

1. Go to <https://github.com/signup>.
2. Enter your email, a password and a **username**. Pick something professional, because it
   appears in your repo link (e.g. `github.com/<username>/cashflow-forecasting`) and on your resume.
3. Solve the puzzle, then type the code GitHub emails you.
4. Choose the **Free** plan. You can skip the personalisation questions.
5. Recommended: *Settings → Password and authentication → Enable two-factor authentication*.

## 2. Create an empty repository

1. Click **+** (top right) → **New repository**.
2. Repository name: `cashflow-forecasting`
3. Description: `Freelancer cash-flow forecasting: LSTM/GRU/N-BEATS vs ETS/Prophet, with conformal prediction intervals and shortfall risk`
4. Choose **Public** (best for your resume) or Private.
5. **Do NOT** tick "Add a README", ".gitignore" or "license". The project already has them,
   and ticking them makes the first upload conflict.
6. Click **Create repository**. Keep this page open.

## 3. Upload the files: pick ONE option

### Option A: browser only (easiest, no installation)

1. On the empty repo page click **"uploading an existing file"**.
2. Open the unzipped `cashflow-forecasting` folder on your computer and make hidden files visible
   (Windows Explorer: *View → Show → Hidden items*; Mac Finder: `Cmd+Shift+.`).
   Select **everything inside it except the `.git` folder** (do include the `.gitignore` file)
   and drag it into the browser window. Folders are kept.
3. Commit message: `Initial commit`, then click **Commit changes**.

> GitHub's web upload accepts at most 100 files per upload and 25 MB per file. This project fits
> (the largest file, `artifacts/nbeats.keras`, is ~14 MB). If the page complains, upload the
> folders in two batches.
> With this option the git history in the zip is not used, which is fine.

### Option B: command line (keeps the commit history; recommended if you know a bit of git)

Install git from <https://git-scm.com/downloads>, then in a terminal:

```bash
cd path/to/cashflow-forecasting
git remote add origin https://github.com/<your-username>/cashflow-forecasting.git
git branch -M main
git push -u origin main
```

When asked for a password, GitHub needs a **personal access token**, not your account password:
*Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token*,
tick the `repo` scope, copy the token and paste it as the password.
(Alternatively install **GitHub Desktop**, <https://desktop.github.com>, and use
*File → Add local repository → Publish repository*, which needs no token.)

Before pushing, set your name on the commit history so it shows as yours:

```bash
git config user.name  "Your Name"
git config user.email "the-email-you-used-on-github@example.com"
git commit --amend --reset-author --no-edit   # re-signs the latest commit as you
```

## 4. Add your friend as a collaborator

1. Your friend creates their own GitHub account (step 1) and tells you their **username**.
2. In your repo: **Settings** (tab at the top of the repo) → **Collaborators** (left menu,
   under "Access") → **Add people**.
3. Type their username or email → select them → **Add to repository**.
4. Your friend gets an email / notification and must click **Accept invitation**
   (also at `https://github.com/<your-username>/cashflow-forecasting/invitations`).
   The invite expires after 7 days.

Collaborators on a personal repository can push code, open and merge pull requests and
manage issues. Only you (the owner) can delete the repo or change its settings.

## 5. Make it look good on your profile (optional, 2 minutes)

* On the repo page, click the ⚙ next to **About** → add *topics*: `time-series`, `forecasting`,
  `deep-learning`, `lstm`, `conformal-prediction`, `tensorflow`, `streamlit`.
* Pin the repo on your profile: profile page → **Customize your pins**.
* Deploy the demo for free on <https://share.streamlit.io>: sign in with GitHub → *New app* →
  pick this repo → main file `app/streamlit_app.py`. You get a public link to put in the README
  and on your resume.

## Working together afterwards

```bash
git clone https://github.com/<your-username>/cashflow-forecasting.git   # your friend, once
git pull                         # get the latest changes before you start
git add -A && git commit -m "describe the change"
git push
```

If you both change the same file at the same time, `git pull` will ask you to merge. For bigger
changes, use a branch (`git checkout -b my-feature`) and open a Pull Request on GitHub.
