# Webcam Setup for Docker

## Prerequisites
- Docker Desktop for macOS with latest updates

## Setting up Webcam Access for Docker on macOS

### Method 1: Using the Privileged Mode (Already Configured)

The `docker-compose.yml` file has been updated to include the following settings:

```yaml
privileged: true
security_opt:
  - seccomp:unconfined
```

These settings should help with device access, but macOS has additional security restrictions.

### Method 2: Manual Camera Permission for Docker Desktop

1. **Allow Camera Access to Docker Desktop**:
   - Go to System Preferences > Security & Privacy > Privacy > Camera
   - Ensure Docker Desktop is in the list and has permissions enabled
   - If Docker Desktop is not in the list, you may need to run the container first, then grant permissions when prompted

### Method 3: Using Browser-Based Webcam Access

Since direct device access can be challenging on macOS, an alternative approach is to:

1. **Ensure Webcam Permissions in Browser**:
   - When you access the Streamlit app at http://localhost:8501, your browser will request camera permission
   - Make sure to grant this permission
   - The camera access will happen through your browser rather than directly through Docker

## Testing the Camera

1. Start the Docker container:
   ```bash
   ./start.sh
   ```

2. Open http://localhost:8501 in your browser

3. Navigate to the Facial Emotion Recognition page in the app

4. When prompted by your browser, grant camera access permissions

5. If the camera still doesn't work, you may need to try running the application outside of Docker for the webcam functionality

## Troubleshooting

If the webcam doesn't work inside Docker on macOS:

1. **Check Browser Permissions**: Make sure your browser has permission to access the camera

2. **Restart Docker Desktop**: Sometimes a full restart of Docker can help

3. **Run Outside Docker**: As a last resort, you can run the Streamlit app directly on your host machine:
   ```bash
   cd /Users/fakhrulfauzi/Documents/Projects/FER\ 2
   streamlit run src/dashboard_app.py
   ```

## Note for Different Operating Systems

- **Linux**: On Linux, you would typically pass the video device (`/dev/video0`) directly to the container
- **Windows**: Similar to macOS, you may need to grant permissions via the Windows Security settings
