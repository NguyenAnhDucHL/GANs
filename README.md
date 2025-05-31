# GANs - Monet Style Transfer

This project implements the creation of artworks in Claude Monet's style using Generative Adversarial Networks (GANs).

## 📖 Description

This is a GANs implementation designed to learn and recreate the painting style of Claude Monet - one of the most famous Impressionist artists. The model uses Deep Convolutional GANs (DCGANs) architecture to generate new paintings with distinctive Impressionist artistic features.

## 🎨 Key Features

- **Style Transfer**: Convert ordinary images into Monet-style artworks
- **Deep Convolutional GANs**: Uses advanced DCGAN architecture
- **Progressive Training**: Step-by-step training with dynamic parameter adjustment
- **Quality Assessment**: Automatic result quality evaluation system
- **Visualization**: Real-time training process and results display

## 🏗️ Model Architecture

### Generator
- **Input**: Random noise vector or original image
- **Architecture**: Deep Convolutional layers with BatchNorm and LeakyReLU
- **Output**: RGB 256x256 pixels image in Monet style

### Discriminator  
- **Input**: Real image (Monet) or fake image (from Generator)
- **Architecture**: Convolutional layers with Dropout
- **Output**: Probability of image being real or fake

## 📊 Training Results

From the final training output:
- **Total steps**: 100 steps
- **Training time**: 35.3 minutes (2117.3 seconds)
- **Average time per step**: 21.2 seconds
- **Best generator loss**: 6.234
- **Quality assessment**: 🔶 Fair - Some Monet characteristics have emerged

## 🚀 Installation and Usage

### System Requirements
```bash
Python 3.7+
TensorFlow 2.x
NumPy
Matplotlib
Pillow (PIL)
Jupyter Notebook
```

### Install Dependencies
```bash
pip install tensorflow numpy matplotlib pillow jupyter
```

### Run the Project
1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the file `notebook40e5ac2f75.ipynb`

3. Run cells in order to:
   - Load and preprocess data
   - Define model architecture
   - Train GANs
   - Visualize results

## 📁 Project Structure

```
GANs/
├── README.md                    # This file
├── notebook40e5ac2f75.ipynb    # Main Jupyter notebook containing implementation
├── .git/                       # Git repository
└── .DS_Store                   # System file (macOS)
```

## 🎯 How to Use

### 1. Training from Scratch
- Run the entire notebook to train a new model
- Adjust hyperparameters in corresponding cells
- Monitor training progress through visualization

### 2. Generate New Images
- Load pretrained model (if available)
- Input random noise or original image
- Generate and save results

### 3. Fine-tuning
- Load checkpoint from previous training
- Continue training with lower learning rate
- Evaluate and compare results

## 🔧 Main Hyperparameters

- **Learning rate**: Adjusts learning speed
- **Batch size**: Number of images in each batch
- **Beta1, Beta2**: Adam optimizer parameters
- **Lambda**: Loss functions weight
- **Epochs**: Number of training iterations

## 📈 Training Monitoring

The model provides metrics for monitoring:
- **Generator Loss**: Generator error rate
- **Discriminator Loss**: Discriminator error rate  
- **Quality Score**: Overall quality assessment
- **Sample Images**: Sample images through epochs

## 🎨 Example Results

Training has shown the model can create artworks with:
- ✅ Characteristic colors of Impressionist style
- ✅ Soft brush strokes like Monet
- ✅ Similar composition and lighting
- 🔄 Needs improvement: Detail and sharpness

## 🔬 Evaluation Methods

The model uses a multi-level evaluation system:
- **🔴 Poor**: Cannot recognize Monet's style
- **🔶 Fair**: Some Monet characteristics appear  
- **🟡 Good**: Clearly shows Monet's style
- **🟢 Excellent**: Quality almost like real artwork

## 🛠️ Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size
2. **Unstable Training**: Adjust learning rate
3. **Mode Collapse**: Change architecture or loss function
4. **Poor Quality**: Increase epochs or improve data

### Tips for Improvement:
- Use larger dataset
- More diverse data augmentation
- Fine-tune hyperparameters
- Try different loss functions

## 📚 References

- [Generative Adversarial Networks (Goodfellow et al.)](https://arxiv.org/abs/1406.2661)
- [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

## 🤝 Contributing

All contributions are welcome! Please:
1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact

If you have questions or suggestions, please create an issue or contact directly.

---

**Happy GANs Training!** 🎨✨ 