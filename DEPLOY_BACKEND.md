# Backend Deployment Guide

## Quick Deploy to Render (Recommended - Free Tier Available)

### Step 1: Prepare Backend for Deployment

Your backend is already configured! The code will automatically:
- Download your model from Hugging Face on first startup
- Install all dependencies from requirements.txt

### Step 2: Deploy to Render

1. **Go to Render**: https://render.com (sign up/login with GitHub)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select repository: `now_you_see_me_backend`
   - Click "Connect"

3. **Configure Service**:
   - **Name**: `resnet50-backend` (or your choice)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free` (512MB RAM)

4. **Add Environment Variables** (Optional):
   - None required for basic setup
   - Model downloads automatically from Hugging Face

5. **Add gunicorn to requirements.txt** (needed for deployment):
   ```bash
   cd backend
   echo "gunicorn==21.2.0" >> requirements.txt
   git add requirements.txt
   git commit -m "Add gunicorn for deployment"
   git push
   ```

6. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your backend URL will be: `https://resnet50-backend.onrender.com`

### Step 3: Update Frontend with Backend URL

1. **Go to your frontend GitHub repo**: `now_you_see_me_frontend`

2. **Add Repository Secret**:
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `BACKEND_API_URL`
   - Value: `https://YOUR-APP-NAME.onrender.com` (your Render URL)
   - Click "Add secret"

3. **Re-run GitHub Action**:
   - Go to Actions tab
   - Click "Deploy Next.js to GitHub Pages"
   - Click "Run workflow" → "Run workflow"

### Step 4: Update Backend CORS

The backend needs to allow requests from your GitHub Pages domain. I'll help you update this.

---

## Alternative: Deploy to Railway

1. **Go to Railway**: https://railway.app
2. **New Project** → "Deploy from GitHub repo"
3. **Select**: `now_you_see_me_backend`
4. **Settings**:
   - Start Command: `gunicorn app:app`
   - Add PORT environment variable (Railway sets this automatically)
5. **Deploy** - Get your URL like `https://your-app.up.railway.app`

---

## Next Steps

After backend is deployed:
1. Get your backend URL
2. Update `BACKEND_API_URL` secret in frontend repo
3. Re-deploy frontend
4. Your app will be live! 🚀

**Let me know when you've deployed the backend and I'll help you update the CORS settings!**
