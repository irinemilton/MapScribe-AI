# 🚀 MapScribe AI - Render Deployment Guide

This guide will help you deploy MapScribe AI to Render's free tier.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be pushed to GitHub
2. **Render Account**: Sign up at [render.com](https://render.com)

## 🔧 Deployment Steps

### 1. Prepare Your Repository

Ensure these files are in your repository:
- ✅ `app.py` - Production entry point
- ✅ `requirements-render.txt` - Optimized dependencies
- ✅ `Procfile` - Process configuration
- ✅ `render.yaml` - Render configuration
- ✅ `render-build.sh` - Build script
- ✅ `render_processor.py` - Lightweight processor

### 2. Deploy on Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → **"Web Service"**
3. **Connect GitHub**: Authorize Render to access your repository
4. **Select Repository**: Choose `MapScribe-AI`
5. **Configure Service**:
   - **Name**: `mapscribe-ai`
   - **Environment**: `Python 3`
   - **Build Command**: `./render-build.sh`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: `Free`

### 3. Environment Variables

Set these in Render dashboard:
```
FLASK_ENV=production
PYTHON_VERSION=3.11.4
```

### 4. Advanced Settings

- **Health Check Path**: `/api/status`
- **Auto-Deploy**: `Yes` (deploys on git push)

## 📊 What's Deployed

### ✅ Features Available on Render:
- 🌐 **Web Interface**: Full Flask web application
- 📤 **File Upload**: Image upload and processing
- 🎯 **Demo Mode**: Simulated results for showcase
- 📊 **API Endpoints**: Status and processing endpoints
- 📱 **Responsive Design**: Mobile-friendly interface

### ⚠️ Limitations on Free Tier:
- **Processing**: Uses lightweight demo processor
- **Storage**: Temporary file storage (files deleted on restart)
- **Memory**: Limited to 512MB RAM
- **Sleep**: Service sleeps after 15 minutes of inactivity

## 🔄 Deployment Process

1. **Build Phase** (~5-10 minutes):
   ```bash
   # System dependencies installation
   apt-get update && apt-get install dependencies
   
   # Python dependencies
   pip install -r requirements-render.txt
   
   # Directory setup
   mkdir -p static/{uploads,results,demo}
   ```

2. **Start Phase** (~30 seconds):
   ```bash
   gunicorn app:app
   ```

3. **Health Check**:
   ```
   GET /api/status
   Expected: {"status": "online", ...}
   ```

## 🌐 Access Your Deployment

After successful deployment:
- **URL**: `https://mapscribe-ai-[random].onrender.com`
- **Status**: `https://your-url.onrender.com/api/status`
- **Demo**: `https://your-url.onrender.com/demo`

## 🐛 Troubleshooting

### Build Failures
```bash
# Check build logs in Render dashboard
# Common issues:
- Missing system dependencies
- Python version conflicts
- Memory limits during build
```

### Runtime Issues
```bash
# Check application logs
# Common issues:
- Import errors
- File permission issues
- Memory limits during processing
```

### Performance Issues
```bash
# Free tier limitations:
- Cold starts (15+ seconds)
- Memory limits (512MB)
- CPU throttling
```

## 🚀 Optimization Tips

### 1. Reduce Cold Starts
- Use a service like UptimeRobot to ping your app
- Implement a lightweight health check

### 2. Optimize Dependencies
```python
# Use lightweight alternatives:
opencv-python-headless  # Instead of opencv-python
pillow                  # Optimized image processing
```

### 3. Efficient File Handling
```python
# Clean up temporary files
import tempfile
import os

def cleanup_temp_files():
    # Remove old uploads
    for file in os.listdir('static/uploads'):
        if file.endswith(('.jpg', '.png')):
            os.remove(f'static/uploads/{file}')
```

## 📈 Upgrading to Paid Plans

For production use, consider upgrading:

### Starter Plan ($7/month):
- No sleep
- More memory (1GB)
- Better performance

### Professional Plan ($25/month):
- Even more resources
- Custom domains
- Advanced features

## 🔗 Useful Links

- **Render Docs**: https://render.com/docs
- **Python on Render**: https://render.com/docs/deploy-flask
- **GitHub Integration**: https://render.com/docs/github

## 📞 Support

If you encounter issues:
1. Check Render dashboard logs
2. Review this deployment guide
3. Check GitHub issues
4. Contact Render support

---

**Happy Deploying!** 🎉

Your MapScribe AI will be live and accessible worldwide once deployed!