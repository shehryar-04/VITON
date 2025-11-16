'use client';

import { useState } from 'react';
import Image from 'next/image';

interface ImageUploadProps {
  productId: number;
  userId: string;
}

export function ImageUpload({ productId, userId }: ImageUploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string>();

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('image', e.target.files[0]);
    formData.append('productId', productId.toString());
    formData.append('userId', userId);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setUploadedImageUrl(data.imagePath);
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <input
        type="file"
        accept="image/*"
        onChange={handleUpload}
        disabled={isUploading}
        className="hidden"
        id="imageUpload"
      />
      <label
        htmlFor="imageUpload"
        className="bg-blue-500 text-white px-4 py-2 rounded cursor-pointer"
      >
        {isUploading ? 'Uploading...' : 'Upload Your Photo'}
      </label>
      
      {uploadedImageUrl && (
        <div className="relative w-64 h-96">
          <Image
            src={uploadedImageUrl}
            alt="Uploaded preview"
            fill
            className="object-contain"
          />
        </div>
      )}
    </div>
  );
}