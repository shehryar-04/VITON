import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import path from 'path';
import fs from 'fs';

export async function POST(req: NextRequest) {
  try {
    console.log('API Route: Starting to process request');
    
    // Parse the request body
    const data = await req.json();
    console.log('API Route: Received data:', data);
    
    if (!data.userImagePath || !data.productImagePath || !data.resultImagePath) {
      console.error('API Route: Missing required fields');
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }

    const { userImagePath, productImagePath, resultImagePath } = data;
    
    // Create a timestamp for the entry
    const timestamp = new Date().toISOString();
    
    // Create a structured entry
    const entry = {
      timestamp,
      userImage: userImagePath,
      productImage: productImagePath,
      resultImage: resultImagePath
    };
    
    console.log('API Route: Writing entry:', entry);
    
    // Convert to JSON string and add newline
    const content = JSON.stringify(entry) + '\n';
    
    // Get the file path
    const filePath = path.join(process.cwd(), 'path.txt');
    console.log('API Route: File path:', filePath);
    
    // Ensure the file exists
    try {
      await fs.promises.access(filePath);
    } catch {
      console.log('API Route: Creating new file');
      await fs.promises.writeFile(filePath, '');
    }
    
    // Write to file
    await writeFile(filePath, content, { flag: 'a' });
    console.log('API Route: Successfully wrote to file');
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('API Route: Error writing paths to file:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to write paths' },
      { status: 500 }
    );
  }
} 