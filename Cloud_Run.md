# Deploying FastAPI to Google Cloud Run

This beginner-friendly guide will walk you through deploying your FastAPI application to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**
   - Have an active Google Cloud account
   - Create a new project or select an existing one
   - Enable billing for your project

2. **Local Development Environment**
   - Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - Install [Docker](https://docs.docker.com/get-docker/)

## Project Setup

1. **Initialize Google Cloud SDK**
   ```bash
   # Login to Google Cloud
   gcloud auth login

   # Set your project ID
   gcloud config set project YOUR_PROJECT_ID
   
   # Configure Docker to use Google Cloud credentials
   gcloud auth configure-docker

   # Enable required services
   gcloud services enable run.googleapis.com containerregistry.googleapis.com
   ```

2. **Project Structure**
   ```
   your-project/
   ├── api/
   │   └── main.py          # FastAPI application
   ├── requirements.txt     # Python dependencies
   ├── Dockerfile          # Container configuration
   └── .dockerignore       # Files to exclude from build
   ```

## Configuration Files

### 1. Dockerfile
```dockerfile
# Use Python 3.9 slim image as base with specific platform
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./api ./api

# Set environment variables
ENV PORT=8080

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 2. .dockerignore
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
*.log
.git
.gitignore
.DS_Store
```

### 3. Requirements
```
fastapi
uvicorn
```

## Deployment Steps

1. **Build the Docker Image**
   ```bash
   # Build and tag your image (make sure you're in the project directory)
   docker build --platform=linux/amd64 -t gcr.io/YOUR_PROJECT_ID/fastapi-app .
   ```

2. **Test Locally (Optional but Recommended)**
   ```bash
   # Run the container locally
   docker run -p 8080:8080 gcr.io/YOUR_PROJECT_ID/fastapi-app
   ```
   Visit http://localhost:8080/docs to verify the API works.

3. **Push to Google Container Registry**
   ```bash
   # Push the image
   docker push gcr.io/YOUR_PROJECT_ID/fastapi-app
   ```

4. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy fastapi-service \
     --image gcr.io/YOUR_PROJECT_ID/fastapi-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```
   You'll be prompted to:
   - Confirm/change the service name
   - Select a region
   - Allow unauthenticated access (type 'y' for public access)

## Managing Environment Variables

### For Development (Local Testing)
1. Create a .env file (add to .gitignore):
```env
API_KEY=your_api_key
DATABASE_URL=your_db_url
ENVIRONMENT=development
```

2. Test locally with env file:
```bash
docker run --env-file .env -p 8080:8080 gcr.io/YOUR_PROJECT_ID/fastapi-app
```

### For Production (Cloud Run)
Set environment variables during deployment:
```bash
gcloud run deploy fastapi-service \
  --image gcr.io/YOUR_PROJECT_ID/fastapi-app \
  --set-env-vars "API_KEY=prod_key,ENVIRONMENT=production"
```

### For Sensitive Data
Use Google Cloud Secret Manager:
1. Create a secret:
```bash
echo -n "your-secret-value" | \
gcloud secrets create my-secret --data-file=-
```

2. Use in deployment:
```bash
gcloud run deploy fastapi-service \
  --image gcr.io/YOUR_PROJECT_ID/fastapi-app \
  --set-secrets="MY_SECRET=my-secret:latest"
```

## Verification and Testing

1. **Get Your Service URL**
   ```bash
   gcloud run services describe fastapi-service \
     --platform managed \
     --region us-central1 \
     --format 'value(status.url)'
   ```

2. **Test Endpoints**
   ```bash
   # Test health check
   curl https://YOUR_SERVICE_URL/health

   # Test root endpoint
   curl https://YOUR_SERVICE_URL/
   ```

3. **View Logs**
   ```bash
   # View recent logs
   gcloud logging read "resource.type=cloud_run_revision" --limit=50
   ```

## Common Issues and Solutions

1. **Container Won't Start**
   - Check your Dockerfile platform specification
   - Verify port configuration (must use PORT from env)
   - Check logs for startup errors

2. **Authentication Issues**
   ```bash
   # Re-authenticate if needed
   gcloud auth login
   gcloud auth configure-docker
   ```

3. **Permission Errors**
   - Verify Cloud Run API is enabled
   - Check service account permissions
   - Ensure billing is enabled

4. **Memory/Performance Issues**
   ```bash
   # Adjust resources if needed
   gcloud run deploy fastapi-service \
     --image gcr.io/YOUR_PROJECT_ID/fastapi-app \
     --memory 512Mi \
     --cpu 1
   ```

## Best Practices for Beginners

1. **Security**
   - Never commit sensitive data or .env files
   - Use Secret Manager for API keys
   - Enable HTTPS for all communications

2. **Monitoring**
   - Check logs regularly
   - Monitor resource usage
   - Set up error alerts

3. **Cost Management**
   - Start with minimum instances
   - Monitor billing regularly
   - Set billing alerts
   - Use the [pricing calculator](https://cloud.google.com/products/calculator)

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Cloud SDK Reference](https://cloud.google.com/sdk/gcloud/reference)
