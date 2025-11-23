from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import os
import json
from pathlib import Path
import cv2
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ImageNet class names
IMAGENET_CLASSES = None

def load_imagenet_classes():
    """Load ImageNet class names from file"""
    global IMAGENET_CLASSES
    IMAGENET_CLASSES = {}
    
    # Try to load from imagenet_classes.txt file
    classes_file = Path(__file__).parent / 'imagenet_classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for idx, line in enumerate(f):
                IMAGENET_CLASSES[idx] = line.strip()
        print(f"✅ Loaded {len(IMAGENET_CLASSES)} ImageNet class names from file")
    else:
        # Fallback to generic names
        print("⚠️ imagenet_classes.txt not found, using generic names")
        for i in range(1000):
            IMAGENET_CLASSES[i] = f'class_{i}'
    
    return IMAGENET_CLASSES

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
model = None
activations = {}
gradients = {}
layer_names = []

def save_gradient(name):
    """Hook to save gradients during backward pass"""
    def hook(grad):
        gradients[name] = grad
    return hook

def generate_gradcam(image_tensor, target_class=None):
    """Generate Grad-CAM heatmap for the input image"""
    global activations, gradients
    
    if model is None:
        return None
    
    # Enable gradients for input
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    
    # Get the last convolutional layer (layer4 in ResNet50)
    target_layer = model.layer4
    
    # Clear previous gradients and activations
    gradients = {}
    activations = {}
    model.zero_grad()
    
    # Forward pass with gradient tracking
    output = model(image_tensor)
    
    # Check if layer4 activation was captured
    if 'layer4' not in activations:
        return None
    
    # Get target class (use predicted class if not specified)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Get the score for target class
    class_score = output[0, target_class]
    
    # Backward pass to compute gradients
    model.zero_grad()
    class_score.backward(retain_graph=True)
    
    # Get gradients and activations
    layer4_activation = activations['layer4']
    
    # Get gradients from the layer4 activation
    gradients_val = layer4_activation.grad
    
    if gradients_val is None:
        # Try alternative method: compute gradients manually
        gradients_val = torch.autograd.grad(class_score, layer4_activation, retain_graph=True)[0]
    
    if gradients_val is None:
        return None
    
    gradients_val = gradients_val.cpu().data.numpy()[0]
    activations_val = layer4_activation.cpu().data.numpy()[0]
    
    # Calculate weights (global average pooling of gradients)
    weights = np.mean(gradients_val, axis=(1, 2))
    
    # Create weighted combination of activation maps
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i]
    
    # Apply ReLU (only positive influences)
    cam = np.maximum(cam, 0)
    
    # Normalize to 0-1
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam

def apply_gradcam_overlay(original_image, cam, alpha=0.4):
    """Apply Grad-CAM heatmap overlay on original image"""
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
    
    # Convert to heatmap (0-255)
    heatmap = np.uint8(255 * cam_resized)
    
    # Apply colormap (COLORMAP_JET: blue=low, red=high)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert original image to numpy array
    img_array = np.array(original_image.convert('RGB'))
    
    # Blend images
    overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    
    # Convert back to PIL Image
    overlayed_image = Image.fromarray(overlayed)
    
    return overlayed_image

def load_model():
    """Load ResNet50 model with checkpoint if available"""
    global model
    
    # Load ResNet50 architecture
    model = models.resnet50(pretrained=False)
    
    # Try to download model from Hugging Face if not present locally
    checkpoint_path = Path(__file__).parent / 'latest_checkpoint.pth'
    
    if not checkpoint_path.exists():
        try:
            print("📥 Downloading model from Hugging Face...")
            downloaded_path = hf_hub_download(
                repo_id="pragsyy1729/now_you_see_me",
                filename="latest_checkpoint.pth",
                cache_dir=str(Path(__file__).parent)
            )
            # Move to expected location
            import shutil
            shutil.copy(downloaded_path, checkpoint_path)
            print("✅ Model downloaded successfully from Hugging Face")
        except Exception as e:
            print(f"⚠️  Could not download from Hugging Face: {e}")
    
    # Try to load checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✅ Loaded custom checkpoint successfully")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("Using pretrained ImageNet weights instead")
            model = models.resnet50(pretrained=True)
    else:
        print("No checkpoint found, using pretrained ImageNet weights")
        model = models.resnet50(pretrained=True)
    
    model.eval()
    return model

def register_hooks():
    """Register forward hooks to capture layer activations"""
    global activations, layer_names
    
    activations = {}
    layer_names = []
    
    def get_activation(name, detach=True):
        def hook(model, input, output):
            if detach:
                activations[name] = output.detach()
            else:
                # Keep gradients for layer4 (needed for Grad-CAM)
                activations[name] = output
                if output.requires_grad:
                    output.retain_grad()
        return hook
    
    # Register hooks for key layers
    layers_to_hook = {
        'conv1': model.conv1,
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4,  # Keep gradients for Grad-CAM
        'avgpool': model.avgpool,
    }
    
    for name, layer in layers_to_hook.items():
        # Don't detach layer4 so we can use it for Grad-CAM
        detach = (name != 'layer4')
        layer.register_forward_hook(get_activation(name, detach=detach))
        layer_names.append(name)
    
    return layer_names

def preprocess_image(image_bytes):
    """Preprocess image for ResNet50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def activation_to_heatmap(activation, num_samples=16):
    """Convert activation tensor to heatmap data"""
    # activation shape: [batch, channels, height, width]
    if len(activation.shape) == 4:
        batch, channels, height, width = activation.shape
        
        # Special handling for avgpool (1x1 spatial size)
        if height == 1 and width == 1:
            # For avgpool, create a nice grid visualization
            values = activation[0, :, 0, 0].cpu().numpy()
            
            # Reshape into a square-ish grid
            grid_size = int(np.ceil(np.sqrt(len(values))))
            padded_values = np.zeros(grid_size * grid_size)
            padded_values[:len(values)] = values
            grid = padded_values.reshape(grid_size, grid_size)
            
            # Normalize
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
            
            return {
                'type': 'grid',
                'shape': [grid_size, grid_size],
                'data': grid.tolist(),
                'num_channels': channels,
                'original_channels': channels
            }
        
        # For all other conv layers with spatial dimensions, create channel grid visualization
        if height > 1 and width > 1:
            return create_channel_grid_visualization(activation)
        
        # Sample channels for regular conv layers (fallback)
        sample_indices = np.linspace(0, channels-1, min(num_samples, channels), dtype=int)
        sampled_activations = activation[0, sample_indices, :, :].cpu().numpy()
        
        # Normalize each channel
        heatmaps = []
        for channel_act in sampled_activations:
            normalized = (channel_act - channel_act.min()) / (channel_act.max() - channel_act.min() + 1e-8)
            heatmaps.append(normalized.tolist())
        
        return {
            'type': '2d',
            'shape': [len(sample_indices), height, width],
            'data': heatmaps,
            'num_channels': channels,
            'sampled_channels': sample_indices.tolist()
        }
    elif len(activation.shape) == 2:
        # Fully connected layer
        values = activation[0].cpu().numpy()
        return {
            'type': '1d',
            'shape': list(values.shape),
            'data': values.tolist(),
            'num_channels': values.shape[0]
        }
    else:
        return None

def create_channel_grid_visualization(activation):
    """Create a grid visualization of all channel activations"""
    batch, channels, height, width = activation.shape
    
    # Get all channel activations
    channel_activations = activation[0].cpu().numpy()
    
    # Determine grid size
    grid_cols = int(np.ceil(np.sqrt(channels)))
    grid_rows = int(np.ceil(channels / grid_cols))
    
    # Create a large grid to hold all channel heatmaps
    grid_height = grid_rows * height
    grid_width = grid_cols * width
    mega_grid = np.zeros((grid_height, grid_width))
    
    # Place each channel's heatmap into the grid
    for idx in range(channels):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # Get and normalize this channel's activation
        channel_map = channel_activations[idx]
        if channel_map.max() > channel_map.min():
            channel_map = (channel_map - channel_map.min()) / (channel_map.max() - channel_map.min())
        
        # Place in grid
        start_row = row * height
        start_col = col * width
        mega_grid[start_row:start_row+height, start_col:start_col+width] = channel_map
    
    return {
        'type': 'channel_grid',
        'shape': [grid_height, grid_width],
        'data': mega_grid.tolist(),
        'num_channels': channels,
        'grid_layout': {
            'rows': grid_rows,
            'cols': grid_cols,
            'cell_height': height,
            'cell_width': width
        }
    }

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model architecture information"""
    if model is None:
        load_model()
        register_hooks()
    
    layers_info = []
    
    # Define layer information with 3D positions
    layer_configs = [
        {'name': 'conv1', 'title': 'Conv Block 1', 'description': 'Low-level edges', 'channels': 64, 'size': 112, 'position': [0, 0, 0]},
        {'name': 'layer1', 'title': 'Conv Block 2', 'description': 'Textures and patterns', 'channels': 256, 'size': 56, 'position': [3, 0, 0]},
        {'name': 'layer2', 'title': 'Conv Block 3', 'description': 'Complex motifs', 'channels': 512, 'size': 28, 'position': [6, 0, 0]},
        {'name': 'layer3', 'title': 'Conv Block 4', 'description': 'Object parts', 'channels': 1024, 'size': 14, 'position': [9, 0, 0]},
        {'name': 'layer4', 'title': 'Conv Block 5', 'description': 'High-level objects', 'channels': 2048, 'size': 7, 'position': [12, 0, 0]},
        {'name': 'avgpool', 'title': 'Global Avg Pool', 'description': 'Final activations', 'channels': 2048, 'size': 1, 'position': [15, 0, 0]},
    ]
    
    return jsonify({
        'architecture': 'ResNet50',
        'layers': layer_configs,
        'total_layers': len(layer_configs)
    })

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """Process uploaded image and return activations"""
    global activations
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Read and preprocess image
        image_file = request.files['image']
        image_bytes = image_file.read()
        original_image, image_tensor = preprocess_image(image_bytes)
        
        # Run forward pass (keep gradients for Grad-CAM)
        activations = {}
        output = model(image_tensor)
        
        # Get predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Get class names
        if IMAGENET_CLASSES is None:
            load_imagenet_classes()
        
        top5_names = [IMAGENET_CLASSES.get(idx.item(), f'class_{idx.item()}') for idx in top5_indices]
        
        # Generate Grad-CAM for top prediction
        gradcam_image_b64 = None
        try:
            # Use the existing forward pass result
            if 'layer4' in activations and activations['layer4'].requires_grad:
                layer4_activation = activations['layer4']
                target_class = top5_indices[0].item()
                
                # Get the score for target class
                class_score = output[0, target_class]
                
                # Backward pass to compute gradients
                model.zero_grad()
                class_score.backward(retain_graph=True)
                
                # Get gradients
                gradients_val = layer4_activation.grad
                if gradients_val is not None:
                    gradients_val = gradients_val.cpu().data.numpy()[0]
                    activations_val = layer4_activation.detach().cpu().data.numpy()[0]
                    
                    # Calculate weights (global average pooling of gradients)
                    weights = np.mean(gradients_val, axis=(1, 2))
                    
                    # Create weighted combination of activation maps
                    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
                    for i, w in enumerate(weights):
                        cam += w * activations_val[i]
                    
                    # Apply ReLU (only positive influences)
                    cam = np.maximum(cam, 0)
                    
                    # Normalize to 0-1
                    if cam.max() > 0:
                        cam = cam / cam.max()
                        
                        # Create overlay image
                        overlayed = apply_gradcam_overlay(original_image, cam)
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        overlayed.save(buffered, format="JPEG")
                        gradcam_image_b64 = f'data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}'
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Convert activations to serializable format
        activation_data = {}
        for layer_name in layer_names:
            if layer_name in activations:
                # Detach before converting (layer4 might still have gradients)
                act = activations[layer_name]
                if isinstance(act, torch.Tensor) and act.requires_grad:
                    act = act.detach()
                activation_data[layer_name] = activation_to_heatmap(act)
        
        # Convert image to base64 for preview
        buffered = io.BytesIO()
        original_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response_data = {
            'success': True,
            'image_preview': f'data:image/jpeg;base64,{img_str}',
            'activations': activation_data,
            'predictions': {
                'top5_indices': top5_indices.tolist(),
                'top5_probabilities': top5_prob.tolist(),
                'top5_names': top5_names
            }
        }
        
        # Add Grad-CAM if available
        if gradcam_image_b64:
            response_data['gradcam_overlay'] = gradcam_image_b64
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/layer-activations/<layer_name>', methods=['GET'])
def get_layer_activation(layer_name):
    """Get specific layer activation"""
    if layer_name not in activations:
        return jsonify({'error': f'Layer {layer_name} not found'}), 404
    
    activation_data = activation_to_heatmap(activations[layer_name])
    return jsonify({
        'layer_name': layer_name,
        'activation': activation_data
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with status information"""
    return jsonify({
        'service': 'ResNet50 3D Visualization Backend',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/api/health',
            'model_info': '/api/model-info',
            'process_image': '/api/process-image (POST)'
        }
    })

# Initialize model when app starts (for production servers like Gunicorn)
def init_app():
    """Initialize the application - load classes but defer model loading"""
    print("🚀 Starting ResNet50 Visualization Backend...")
    load_imagenet_classes()
    print("✅ Backend initialized! Model will load on first request.")

def ensure_model_loaded():
    """Lazy load model on first request to save memory during startup"""
    global model
    if model is None:
        print("📦 Loading model on first request...")
        load_model()
        register_hooks()
        print("✅ Model loaded successfully!")

# Call init when module is imported (works with Gunicorn)
init_app()

if __name__ == '__main__':
    print("🌐 Backend running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
