import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

const WebcamView = forwardRef(function WebcamView({ isStreaming, onPermissionError }, ref) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        return "";
      }
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        return "";
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.85);
    },
  }));

  useEffect(() => {
    if (!isStreaming) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      return;
    }

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch {
        onPermissionError("Kamera izni olmadan yayın başlatılamadı.");
      }
    };

    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      streamRef.current = null;
    };
  }, [isStreaming, onPermissionError]);

  return (
    <div className="overflow-hidden rounded-2xl bg-slate-900/75 shadow-soft ring-1 ring-slate-700">
      <video ref={videoRef} className="aspect-video w-full object-cover" playsInline muted />
      {!isStreaming && (
        <div className="flex aspect-video items-center justify-center bg-slate-950 text-slate-400">
          Kamera kapalı
        </div>
      )}
    </div>
  );
});

export default WebcamView;
