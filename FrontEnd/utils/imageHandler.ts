import sharp from 'sharp';
import path from 'path';
import fs from 'fs/promises';

interface ProcessImageParams {
  userImage: Buffer;
  productId: number;
  userId: string;
  clothingImage: string; // Path to the clothing image from categories folder
}

export class ImageHandler {
  private readonly uploadDir = path.join(process.cwd(), 'public/images/try');
  private readonly categoriesDir = path.join(process.cwd(), 'public/categories');

  constructor() {
    this.ensureUploadDirectory();
  }

  private async ensureUploadDirectory() {
    try {
      await fs.access(this.uploadDir);
    } catch {
      await fs.mkdir(this.uploadDir, { recursive: true });
    }
  }

  async processAndSaveImages({ userImage, productId, userId, clothingImage }: ProcessImageParams): Promise<string> {
    try {
      // Generate unique filename
      const filename = `${userId}-${productId}-${Date.now()}.png`;
      const outputPath = path.join(this.uploadDir, filename);

      // Get the clothing image path
      const clothingImagePath = path.join(this.categoriesDir, path.basename(clothingImage));

      // Process user image first
      const processedUserImage = await sharp(userImage)
        .resize(800, 1200, { fit: 'contain' })
        .toBuffer();

      // Combine images - place clothing on top of user image with transparency
      await sharp(processedUserImage)
        .composite([
          {
            input: clothingImagePath,
            blend: 'over',
            gravity: 'center'
          }
        ])
        .toFile(outputPath);

      return `/images/try/${filename}`;
    } catch (error) {
      console.error('Error processing images:', error);
      throw new Error('Failed to process images');
    }
  }
}