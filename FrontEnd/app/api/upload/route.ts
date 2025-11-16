import { NextRequest, NextResponse } from 'next/server';
import sharp from 'sharp';
import path from 'path';
import { writeFile } from 'fs/promises';
import fs from 'fs/promises';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('image') as Blob;
    const productId = formData.get('productId') as string;
    const clothingImage = formData.get('clothingImage') as string;

    if (!file || !productId || !clothingImage) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Convert blob to buffer
    const buffer = Buffer.from(await file.arrayBuffer());
    
    // Get the clothing image path from public directory
    const clothingImagePath = path.join(process.cwd(), 'public', clothingImage);

    // Ensure clothing image exists
    try {
      await fs.access(clothingImagePath);
    } catch (error) {
      return NextResponse.json(
        { success: false, error: 'Clothing image not found' },
        { status: 404 }
      );
    }

    // Process user image
    const processedUserImage = await sharp(buffer)
      .resize(800, 1200, { fit: 'contain' })
      .toBuffer();

    // Ensure output directory exists
    const outputDir = path.join(process.cwd(), 'public', 'images', 'try');
    await fs.mkdir(outputDir, { recursive: true });

    // Generate unique filename
    const filename = `try-on-${productId}-${Date.now()}.png`;
    const outputPath = path.join(outputDir, filename);

    // Combine images
    await sharp(processedUserImage)
      .composite([
        {
          input: clothingImagePath,
          blend: 'over',
          gravity: 'center'
        }
      ])
      .toFile(outputPath);

    return NextResponse.json({ 
      success: true,
      imagePath: `/images/try/${filename}` 
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to process image' },
      { status: 500 }
    );
  }
}