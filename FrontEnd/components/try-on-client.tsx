"use client"

import type React from "react"
import type { Product } from "@/lib/static-data"

import { useState, useRef, useEffect } from "react"
import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { useToast } from "@/components/ui/use-toast"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { saveToHistory } from "@/lib/supabase-helpers"
import { Camera, Upload, Save, RotateCw, ArrowLeft, Eye } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"

const writePathsToFile = async (userImagePath: string, productImagePath: string, resultImagePath: string) => {
  try {
    console.log('Starting to write paths to file...');
    console.log('User Image Path:', userImagePath);
    console.log('Product Image Path:', productImagePath);
    console.log('Result Image Path:', resultImagePath);

    const response = await fetch('/api/save-paths', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userImagePath,
        productImagePath,
        resultImagePath,
      }),
    });

    console.log('API Response Status:', response.status);
    const responseData = await response.json();
    console.log('API Response Data:', responseData);

    if (!response.ok) {
      throw new Error(responseData.error || 'Failed to save paths');
    }

    return responseData;
  } catch (error) {
    console.error('Error in writePathsToFile:', error);
    throw error;
  }
};

export default function TryOnClient({ product }: { product: Product }) {
  const { toast } = useToast()
  const [userImage, setUserImage] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState(0) // Added progress state
  const [activeTab, setActiveTab] = useState("upload")
  const [isCapturing, setIsCapturing] = useState(false)
  const [countdownValue, setCountdownValue] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [productImageError, setProductImageError] = useState(false)

  // Clean up camera stream when component unmounts
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
    }
  }, [stream])

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      // Show loading toast
      toast({
        title: "Processing image",
        description: "Your photo is being prepared for try-on",
      });
      
      // Check file size
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        toast({
          title: "File too large",
          description: "Please upload an image smaller than 10MB",
          variant: "destructive",
        });
        return;
      }
      
      const reader = new FileReader()
      reader.onload = (event) => {
        if (event.target?.result) {
          // Store the file data URL for preview
          setUserImage(event.target.result as string)
          // After setting the image, automatically move to the preview tab
          setActiveTab("preview")
          console.log("Image loaded successfully")
          toast({
            title: "Image ready",
            description: "Your photo is ready for try-on",
          });
        }
      }
      reader.onerror = (error) => {
        console.error("Error reading file:", error)
        toast({
          title: "Upload Error",
          description: "There was a problem processing your image. Please try again.",
          variant: "destructive",
        })
      }
      reader.readAsDataURL(file)
    } else {
      console.log("No file selected")
    }
  }

  const startCamera = async () => {
    try {
      toast({
        title: "Accessing camera",
        description: "Please allow camera access when prompted",
      });
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "user", // Prefer front camera
          width: { ideal: 1280 },
          height: { ideal: 720 } 
        } 
      });
      
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        toast({
          title: "Camera active",
          description: "Position yourself in the frame and take a photo",
        });
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
      
      // Provide more helpful error messages based on the error
      let errorMessage = "Could not access your camera. Please check permissions.";
      
      if (error instanceof DOMException) {
        if (error.name === "NotAllowedError") {
          errorMessage = "Camera access denied. Please allow camera access in your browser settings.";
        } else if (error.name === "NotFoundError") {
          errorMessage = "No camera found. Please ensure your device has a working camera.";
        } else if (error.name === "NotReadableError") {
          errorMessage = "Camera is in use by another application. Please close other apps using the camera.";
        } else if (error.name === "OverconstrainedError") {
          errorMessage = "Camera does not meet the required constraints. Try a different camera.";
        } else if (error.name === "AbortError") {
          errorMessage = "Camera operation was aborted. Please try again.";
        }
      }
      
      toast({
        title: "Camera Error",
        description: errorMessage,
        variant: "destructive",
      });
      
      // Switch back to upload tab on error
      setActiveTab("upload");
    }
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
  }

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      if (isCapturing) return; // Prevent multiple captures
      
      setIsCapturing(true);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      // Start a countdown before capturing
      let countdown = 3;
      setCountdownValue(countdown);
      
      toast({
        title: "Get Ready",
        description: `Taking photo in ${countdown} seconds...`,
      });
      
      const countdownInterval = setInterval(() => {
        countdown--;
        setCountdownValue(countdown);
        
        if (countdown > 0) {
          toast({
            title: "Get Ready",
            description: `Taking photo in ${countdown} seconds...`,
          });
        } else {
          clearInterval(countdownInterval);
          
          // Actually take the photo
          const context = canvas.getContext("2d");
          
          if (context) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Flip the image horizontally for a mirror effect (selfie mode)
            context.translate(canvas.width, 0);
            context.scale(-1, 1);
            
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageDataUrl = canvas.toDataURL("image/png");
            setUserImage(imageDataUrl);
            stopCamera();
            setActiveTab("preview");
            
            toast({
              title: "Photo Captured",
              description: "Your photo is ready for try-on",
            });
          }
          
          setIsCapturing(false);
        }
      }, 1000);
    }
  }

  const handleTryOn = async () => {
    if (!userImage) {
      toast({
        title: "No Image",
        description: "Please upload or capture your photo first.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);
    console.log("Starting try-on process...");

    try {
      // Process the try-on
      await simulateVirtualTryOn();
      
      // Note: writePathsToFile is now called only from simulateVirtualTryOn when needed
      // Removed the redundant call from here
    } catch (error) {
      console.error('Error in try-on process:', error);
      toast({
        title: "Error",
        description: "Failed to process try-on",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  // Function to open the image in a new tab
  const openImageInNewTab = () => {
    if (!resultImage) return;
    
    try {
      // Open image in a new tab
      window.open(resultImage, '_blank');
      
      toast({
        title: "Image Opened",
        description: "The image has been opened in a new tab",
      });
    } catch (error) {
      console.error("Error opening image in new tab:", error);
      
      // Try to copy the URL as fallback
      navigator.clipboard.writeText(resultImage)
        .then(() => {
          toast({
            title: "Image URL Copied",
            description: "The image URL has been copied to your clipboard. You can paste it in a new browser tab.",
            duration: 5000,
          });
        })
        .catch(clipboardError => {
          console.error("Clipboard copy failed:", clipboardError);
          toast({
            title: "Error",
            description: "Could not open image or copy URL. Please try saving the image instead.",
            variant: "destructive",
          });
        });
    }
  };

  const simulateVirtualTryOn = async () => {
    try {
      if (!userImage) {
        throw new Error("No user image available")
      }

      // Convert user image data URL to Blob
      const userImageResponse = await fetch(userImage)
      const userImageBlob = await userImageResponse.blob()
      
      // Fetch the cloth image from URL and convert to Blob
      const clothImageUrl = product.tryOnImage || product.image
      console.log("Fetching cloth image from:", clothImageUrl)
      const clothImageResponse = await fetch(clothImageUrl)
      if (!clothImageResponse.ok) {
        throw new Error(`Failed to fetch cloth image: ${clothImageResponse.status}`)
      }
      const clothImageBlob = await clothImageResponse.blob()
      
      // Create form data
      const formData = new FormData()
      formData.append('user_image', userImageBlob, 'user_image.png')
      formData.append('cloth_image', clothImageBlob, 'cloth_image.png')
      formData.append('category_id', product.categorySlug)

      console.log("Sending try-on request to backend...")
      
      // Call FastAPI backend
      const tryOnResponse = await fetch('http://localhost:8000/api/try-on', {
        method: 'POST',
        body: formData
      })

      const data = await tryOnResponse.json()
      console.log("Try-on API response:", data)
      
      if (!tryOnResponse.ok) {
        throw new Error(data.detail || data.error || `Server error: ${tryOnResponse.status}`)
      }

      // Poll for result if status is pending
      if (data.data.status === "pending") {
        // Increase maximum attempts to allow for 2-minute processing
        // 60 attempts × 2 seconds = 120 seconds (2 minutes)
        let attemptCount = 0;
        const MAX_ATTEMPTS = 70; // Set slightly higher than needed (140 seconds)
        
        // Show a notification for long processing
        toast({
          title: "Processing Started",
          description: "This may take up to 2 minutes to complete.",
          duration: 5000,
        });
        
        const pollInterval = setInterval(async () => {
          try {
            attemptCount++;
            console.log(`Polling attempt ${attemptCount}/${MAX_ATTEMPTS} for ID: ${data.data.id}...`);
            
            // Update UI with progress
            setProcessingProgress(Math.min(95, Math.round((attemptCount / MAX_ATTEMPTS) * 100)));
            
            // Wrap status check in try-catch to handle potential errors
            let statusData;
            try {
              const statusResponse = await fetch(`http://localhost:8000/api/try-on/${data.data.id}`)
              statusData = await statusResponse.json()
              console.log("Status update:", statusData);
            } catch (statusError) {
              console.warn(`Error fetching status (attempt ${attemptCount}):`, statusError);
              // Don't fail completely on a single status check error
              if (attemptCount > MAX_ATTEMPTS - 5) {
                throw statusError; // Only throw after multiple consecutive failures
              }
              return; // Skip this polling iteration but continue polling
            }
            
            // Always check for message field which might indicate a temporary error
            if (statusData.data.message && statusData.data.message.includes("unavailable")) {
              console.warn("Status check temporarily unavailable, continuing to poll...");
              // Don't fail, just continue polling
              return;
            }
            
            // Critical check: Always look for result image URL first, regardless of status
            if (statusData.data.result_image_url) {
              clearInterval(pollInterval);
              console.log("Result image URL found:", statusData.data.result_image_url);
              setProcessingProgress(100);
              
              // Force re-render by setting state with a small delay
              setTimeout(() => {
                setResultImage(statusData.data.result_image_url);
                console.log("Result image state updated with:", statusData.data.result_image_url);
              }, 100);
              
              // Save to history
              try {
                console.log("Attempting to save to history:", {
                  product_id: product.id,
                  product_name: product.name,
                  status: "completed"
                });
                
                const result = await saveToHistory({
                  product_id: product.id,
                  product_name: product.name,
                  product_image: product.image,
                  user_image_url: data.data.user_image_url,
                  cloth_image_url: data.data.cloth_image_url,
                  result_image_url: statusData.data.result_image_url,
                  status: "completed"
                });
                
                if (result.success) {
                  console.log("History saved successfully");
                } else {
                  console.error("Error saving history:", result.error);
                  toast({
                    title: "Warning",
                    description: "Try-on completed but couldn't save to history.",
                    variant: "destructive"
                  });
                }
              } catch (historyError) {
                console.error("Error saving history, but processing succeeded:", historyError);
                toast({
                  title: "Warning",
                  description: "Try-on completed but couldn't save to history.",
                  variant: "destructive"
                });
              }
              
              // Write paths to file
              try {
                await writePathsToFile(
                  userImage.startsWith('data:') ? 'In-memory image' : userImage,
                  product.image,
                  statusData.data.result_image_url
                );
              } catch (pathError) {
                console.error("Error writing paths to file, but processing succeeded:", pathError);
              }
              
              return; // Exit early once we have a result
            }
            
            // Status checks
            if (statusData.data.status === "completed") {
              // If status is completed but no image yet, keep polling for a bit longer
              console.log("Status is completed but no image URL yet, continuing to poll...");
              
              // If we've been polling for a while after completion, give up
              if (attemptCount > MAX_ATTEMPTS - 10) {
                clearInterval(pollInterval);
                throw new Error("Process completed but no result image available after extended polling");
              }
            }
            else if (statusData.data.status === "failed") {
              clearInterval(pollInterval);
              throw new Error(statusData.data.error || "Processing failed");
            }
            else if (attemptCount >= MAX_ATTEMPTS) {
              clearInterval(pollInterval);
              throw new Error("Timed out waiting for processing to complete after extended polling");
            }
          } catch (pollError) {
            console.error("Error during polling:", pollError);
            clearInterval(pollInterval);
            
            // Don't immediately fail - check if we already have a result URL from the backend logs
            try {
              const finalCheckResponse = await fetch(`http://localhost:8000/api/try-on/${data.data.id}`);
              const finalData = await finalCheckResponse.json();
              
              if (finalData?.data?.result_image_url) {
                console.log("Found result URL in final check:", finalData.data.result_image_url);
                setResultImage(finalData.data.result_image_url);
                return; // Success despite polling error
              }
            } catch (finalError) {
              // If final check also fails, then throw the original error
              console.error("Final status check also failed:", finalError);
            }
            
            throw pollError;
          }
        }, 2000); // Poll every 2 seconds
      } 
      // If the initial response already includes a result image URL, use it directly
      else if (data.data.result_image_url) {
        console.log("Result image URL already available:", data.data.result_image_url);
        setResultImage(data.data.result_image_url);
        
        // Write paths to file only once when we have the final result
        console.log("Saving path information...");
        await writePathsToFile(
          userImage.startsWith('data:') ? 'In-memory image' : userImage,
          product.image,
          data.data.result_image_url
        );
      }

    } catch (error) {
      console.error("Error in virtual try-on:", error)
      toast({
        title: "Processing Error",
        description: error instanceof Error ? error.message : "Could not process the image. Please try again.",
        variant: "destructive",
      })
    }
  }

  const handleSaveImage = async () => {
    if (!resultImage) {
      toast({
        title: "No image to save",
        description: "Please try on the product first to generate an image",
        variant: "destructive",
      });
      return;
    }
    
    // Generate a consistent filename with product details
    const fileName = `styletry-${product.slug.replace(/\s+/g, '-').toLowerCase()}.jpg`;
    
    try {
      toast({
        title: "Downloading...",
        description: "Preparing your image for download",
      });
      
      // Primary download method: Fetch + Blob approach (most reliable)
      try {
        console.log("Attempting fetch-based download method...");
        
        // Enhanced fetch-based download approach to handle CORS properly
        const response = await fetch(resultImage, {
          method: 'GET',
          mode: 'cors',
          cache: 'no-cache',
        });
        
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }
        
        // Get the image as a blob
        const blob = await response.blob();
        
        // Create a blob URL
        const blobUrl = URL.createObjectURL(blob);
        
        // Create a temporary link element
        const link = document.createElement('a');
        
        // Set link properties for download
        link.href = blobUrl;
        link.download = fileName; // This is critical for forcing download
        link.style.display = 'none';
        
        // Append to document, click, and remove
        document.body.appendChild(link);
        link.click();
        
        // Short delay before removing to ensure browser has time to process
        setTimeout(() => {
          document.body.removeChild(link);
          URL.revokeObjectURL(blobUrl); // Free up memory
        }, 100);
        
        toast({
          title: "Download Complete",
          description: "Your virtual try-on image has been saved",
        });
        
        return; // Exit if direct download works
      } catch (fetchError) {
        console.error("Fetch-based download failed, trying fallback method:", fetchError);
        toast({
          title: "Trying alternative method",
          description: "First download attempt failed, trying another approach...",
        });
      }
      
      // Second attempt: Canvas method with robust CORS handling
      try {
        console.log("Attempting canvas download method...");
        
        // Create an image element to load the result image
        const img = document.createElement('img');
        img.crossOrigin = "anonymous"; // Important for CORS handling
        
        // Create a promise to handle image loading
        await new Promise((resolve, reject) => {
          // Set a timeout to prevent hanging
          const timeout = setTimeout(() => {
            reject(new Error("Image loading timed out"));
          }, 15000); // Extended timeout (15 seconds)
          
          img.onload = () => {
            clearTimeout(timeout);
            resolve(null);
          };
          
          img.onerror = () => {
            clearTimeout(timeout);
            console.error("Image loading error");
            reject(new Error("Failed to load the image for download"));
          };
          
          // Try appending a cache-busting parameter to bypass any caching issues
          const cacheBuster = `?t=${new Date().getTime()}`;
          img.src = resultImage.includes('?') ? resultImage : resultImage + cacheBuster;
        });
        
        // Create a canvas element to draw the image
        const canvas = document.createElement('canvas');
        canvas.width = img.width || 800; // Fallback width if image dimensions aren't available
        canvas.height = img.height || 1000; // Fallback height
        const ctx = canvas.getContext("2d");
        
        if (!ctx) {
          throw new Error("Could not create canvas context");
        }
        
        // Draw the image to the canvas
        ctx.drawImage(img, 0, 0);
        
        // Try to use the Canvas.toBlob method for better browser compatibility
        canvas.toBlob((blob) => {
          if (!blob) {
            throw new Error("Failed to create blob from canvas");
          }
          
          // Create a blob URL
          const blobUrl = URL.createObjectURL(blob);
          
          // Create a temporary link element
          const link = document.createElement('a');
          link.href = blobUrl;
          link.download = fileName;
          link.target = "_blank"; // Ensure it doesn't use the current window
          link.rel = "noopener noreferrer"; // Security best practice
          link.style.display = 'none';
          
          // Append to document, click, and remove
          document.body.appendChild(link);
          link.click();
          
          // Clean up - give the browser a moment to process the download
          setTimeout(() => {
            document.body.removeChild(link);
            URL.revokeObjectURL(blobUrl); // Free up memory
          }, 100);
          
          toast({
            title: "Download Complete",
            description: "Your virtual try-on image has been saved",
          });
        }, 'image/jpeg', 0.95); // Use JPEG with high quality
        
      } catch (canvasError) {
        console.error("Canvas method failed, trying iframe fallback:", canvasError);
        toast({
          title: "Trying final download method",
          description: "Previous methods failed, attempting one last approach...",
        });
        
        // Third attempt: Iframe fallback method
        try {
          console.log("Attempting iframe download method...");
          
          // Create temporary iframe
          const iframe = document.createElement('iframe');
          iframe.style.display = 'none';
          document.body.appendChild(iframe);
          
          // Force download using iframe
          const iframeDoc = iframe.contentWindow?.document;
          if (iframeDoc) {
            iframeDoc.open();
            iframeDoc.write(`
              <!DOCTYPE html>
              <html>
                <head><title>Download</title></head>
                <body>
                  <a href="${resultImage}" download="${fileName}" id="download-link" target="_blank" rel="noopener noreferrer">Download</a>
                  <script>
                    // Force download behavior
                    const downloadLink = document.getElementById('download-link');
                    downloadLink.addEventListener('click', function(e) {
                      // Prevent default navigation
                      e.preventDefault();
                      
                      // Create a blob URL from the image
                      fetch('${resultImage}')
                        .then(response => response.blob())
                        .then(blob => {
                          const blobUrl = URL.createObjectURL(blob);
                          
                          // Create a new link with download attribute
                          const a = document.createElement('a');
                          a.href = blobUrl;
                          a.download = "${fileName}";
                          a.style.display = 'none';
                          document.body.appendChild(a);
                          a.click();
                          document.body.removeChild(a);
                          URL.revokeObjectURL(blobUrl);
                        });
                    });
                    
                    // Trigger click
                    document.getElementById('download-link').click();
                  </script>
                </body>
              </html>
            `);
            iframeDoc.close();
            
            // Give the iframe some time to process the download
            setTimeout(() => {
              document.body.removeChild(iframe);
              
              toast({
                title: "Download Initiated",
                description: "Your browser should prompt you to save the image",
              });
            }, 1000);
          } else {
            throw new Error("Could not create iframe document");
          }
        } catch (iframeError) {
          console.error("All download methods failed:", iframeError);
          
          // Attempt to copy to clipboard as last resort
          try {
            await navigator.clipboard.writeText(resultImage);
            toast({
              title: "Download Failed - Link Copied",
              description: "Could not download automatically. A link to the image has been copied to your clipboard. You can paste it in your browser and save manually.",
              duration: 7000,
            });
          } catch (clipboardError) {
            console.error("Clipboard copy also failed:", clipboardError);
            throw new Error("Could not download the image after multiple attempts");
          }
        }
      }
    } catch (error) {
      console.error("Error saving image:", error);
      
      // Final fallback - provide detailed instructions to the user
      toast({
        title: "Download Failed",
        description: "Could not download automatically. Try right-clicking on the image and selecting 'Save image as...'",
        variant: "destructive",
        duration: 10000,
      });
      
      // Show a second toast with additional help
      setTimeout(() => {
        toast({
          title: "Alternative Methods",
          description: "You can also screenshot the image or open it in a new tab by copying the image URL",
          duration: 8000,
        });
      }, 2000);
    }
  }

  return (
    <div className="container py-10">
      <div className="mb-8">
        <Link
          href={`/products/${product.slug}`}
          className="text-sm text-muted-foreground hover:underline inline-flex items-center"
        >
          <ArrowLeft className="mr-1 h-4 w-4" /> Back to Product
        </Link>
        <h1 className="text-3xl font-bold mt-2">Try On: {product.name}</h1>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <div className="flex flex-col">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="upload" className="flex items-center gap-1">
                <Upload className="h-4 w-4" />
                Upload Photo
              </TabsTrigger>
              <TabsTrigger value="camera" onClick={startCamera} className="flex items-center gap-1">
                <Camera className="h-4 w-4" />
                Use Camera
              </TabsTrigger>
            </TabsList>
            <TabsContent value="upload" className="border rounded-md p-6">
              <div className="flex flex-col items-center justify-center gap-4">
                <p className="text-center text-muted-foreground">
                  Upload a front-facing photo of yourself for the virtual try-on
                </p>
                <div className="flex flex-col gap-4 items-center w-full">
                  <Button onClick={() => fileInputRef.current?.click()} className="gap-2">
                    <Upload className="h-4 w-4" />
                    Select Photo
                  </Button>
                  <p className="text-xs text-muted-foreground">or</p>
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer bg-secondary/20 hover:bg-secondary/30">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-8 h-8 mb-2 text-muted-foreground" />
                      <p className="mb-2 text-sm text-muted-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground">PNG, JPG or JPEG</p>
                    </div>
                    <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                  </label>
                </div>
                <input type="file" accept="image/*" className="hidden" ref={fileInputRef} onChange={handleFileUpload} />
              </div>
            </TabsContent>
            <TabsContent value="camera" className="border rounded-md p-6">
              <div className="flex flex-col items-center justify-center gap-4">
                <p className="text-center text-muted-foreground mb-2">
                  Position yourself in the center of the frame, facing forward
                </p>
                <div className="relative w-full max-w-md rounded-md overflow-hidden border border-gray-300">
                  <video 
                    ref={videoRef} 
                    autoPlay 
                    playsInline 
                    className="w-full max-w-md rounded-md transform scale-x-[-1]" 
                    style={{ transform: 'scaleX(-1)' }} 
                  />
                  <div className="absolute inset-0 pointer-events-none border-2 border-dashed border-primary/50 m-4 rounded"></div>
                  
                  {isCapturing && countdownValue > 0 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                      <span className="text-white font-bold text-6xl">{countdownValue}</span>
                    </div>
                  )}
                </div>
                <canvas ref={canvasRef} className="hidden" />
                <div className="flex gap-2 w-full justify-center">
                  <Button variant="outline" onClick={() => setActiveTab("upload")} className="gap-2">
                    <Upload className="h-4 w-4" />
                    Switch to Upload
                  </Button>
                  <Button 
                    onClick={capturePhoto} 
                    disabled={isCapturing}
                    className="gap-2"
                  >
                    <Camera className="h-4 w-4" />
                    {isCapturing ? `Taking photo in ${countdownValue}...` : "Take Photo"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  For best results, use good lighting and a plain background
                </p>
              </div>
            </TabsContent>
            <TabsContent value="preview" className="border rounded-md p-6">
              <div className="flex flex-col items-center justify-center gap-4">
                {userImage && (
                  <div className="relative">
                    <Image
                      src={userImage || "/placeholder.svg"}
                      alt="Your photo"
                      width={400}
                      height={500}
                      className="rounded-md max-h-[400px] w-auto"
                      unoptimized
                    />
                  </div>
                )}
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => setActiveTab("upload")} className="gap-2">
                    <RotateCw className="h-4 w-4" />
                    Change Photo
                  </Button>
                  <Button onClick={handleTryOn} disabled={isProcessing}>
                    {isProcessing ? "Processing..." : "Try On Now"}
                  </Button>
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {userImage && activeTab === "preview" && (
            <div className="mt-6">
              <h3 className="text-lg font-medium mb-2">Selected Product</h3>
              <div className="flex items-center gap-4">
                <div className="relative w-[100px] h-[130px] rounded-md border overflow-hidden">
                  <Image
                    src={product.image || "/placeholder.svg?height=130&width=100&text=Product"}
                    alt={`Product image: ${product.name}`}
                    fill
                    className="object-cover"
                    onError={() => {
                      setProductImageError(true);
                      toast({
                        title: "Image Error",
                        description: "Could not load product image. Using placeholder instead.",
                        variant: "destructive",
                      });
                    }}
                    unoptimized
                  />
                </div>
                <div>
                  <p className="font-medium">{product.name}</p>
                  <p className="text-primary font-medium">${product.price.toFixed(2)}</p>
                  <p className="text-sm text-muted-foreground mt-1">{product.categorySlug}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Try-On Result</h2>
          <Card className="border">
            <CardContent className="p-6 flex flex-col items-center">
              {resultImage ? (
                <>
                  <div className="relative w-full flex justify-center">
                    <Image
                      key={`result-${resultImage}`}
                      src={resultImage || "/placeholder.svg"}
                      alt={`Virtual try-on result of ${product.name}`}
                      width={400}
                      height={500}
                      className="rounded-md max-h-[500px] w-auto object-contain"
                      unoptimized
                      priority={true}
                      crossOrigin="anonymous"
                      aria-label="Try-on result image"
                      onError={(e) => {
                        console.error("Failed to load result image");
                        toast({
                          title: "Image Error",
                          description: "The result image failed to load. Please try again.",
                          variant: "destructive",
                        });
                      }}
                    />
                  </div>
                  <div className="flex flex-wrap gap-2 mt-4 w-full justify-center">
                    <Button 
                      variant="outline" 
                      onClick={handleSaveImage} 
                      className="gap-2"
                    >
                      <Save className="h-4 w-4" />
                      Save Image
                    </Button>
                    <Button 
                      onClick={() => {
                        // Copy image URL to clipboard as a fallback option
                        navigator.clipboard.writeText(resultImage)
                          .then(() => {
                            toast({
                              title: "Image URL Copied",
                              description: "The image URL has been copied to your clipboard as a backup option",
                            });
                          })
                          .catch(err => {
                            console.error("Failed to copy URL:", err);
                            // Silently continue even if clipboard copy fails
                          });
                        
                        // Reset processing progress
                        setProcessingProgress(0);
                        
                        // Show toast about reprocessing
                        toast({
                          title: "Processing New Try-On",
                          description: "Creating a new version of your virtual try-on...",
                        });
                        
                        // Then trigger the try-on process again
                        handleTryOn();
                      }} 
                      className="gap-2"
                    >
                      <RotateCw className="h-4 w-4" />
                      Try Again
                    </Button>
                    <Button variant="outline" onClick={openImageInNewTab} className="gap-2">
                      <Eye className="h-4 w-4" />
                      Open Image
                    </Button>
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-[400px] text-center">
                  {isProcessing ? (
                    <div className="flex flex-col items-center justify-center">
                      <div 
                        className="w-full bg-gray-200 rounded-full h-2.5 mb-4"
                        role="progressbar" 
                        aria-valuemin={0} 
                        aria-valuemax={100} 
                        aria-valuenow={processingProgress}
                        aria-label="Processing progress"
                      >
                        <div 
                          className="bg-primary h-2.5 rounded-full transition-all duration-300" 
                          style={{ width: `${processingProgress}%` }}
                        ></div>
                      </div>
                      <p className="text-muted-foreground">Processing your try-on request... {processingProgress}%</p>
                      <p className="text-xs text-gray-500 mt-2">This may take up to 2 minutes</p>
                    </div>
                  ) : (
                    <p className="text-muted-foreground mb-4">
                      Upload or capture your photo and click 'Try On Now' to see the result
                    </p>
                  )}
                  {isProcessing && !processingProgress && (
                    <div 
                      className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mt-4"
                      role="status"
                      aria-label="Processing"
                    >
                      <span className="sr-only">Processing...</span>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <div className="mt-8">
            <h3 className="text-lg font-medium mb-4">How It Works</h3>
            <ol className="list-decimal list-inside space-y-2 text-muted-foreground">
              <li>Upload a front-facing photo or use your camera</li>
              <li>Our system analyzes your body position and dimensions</li>
              <li>The selected clothing item is digitally fitted to your image</li>
              <li>View and save the result to share with friends</li>
            </ol>
          </div>
          
          {resultImage && (
            <div className="mt-4 p-4 bg-muted/30 rounded-md">
              <h3 className="text-lg font-medium mb-2">Having trouble downloading?</h3>
              <p className="text-sm text-muted-foreground mb-2">
                If the download button doesn't work, try these alternatives:
              </p>
              <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                <li>Use the "Open Image" button and then right-click to save</li>
                <li>Right-click directly on the image and select "Save image as..."</li>
                <li>Take a screenshot of the image</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
