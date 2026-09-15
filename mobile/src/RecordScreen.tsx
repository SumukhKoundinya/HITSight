import { CameraView, useCameraPermissions, useMicrophonePermissions } from "expo-camera";
import { useEffect, useRef, useState } from "react";
import { Platform, StyleSheet, Text, TouchableOpacity, View } from "react-native";

type Props = {
  durationSec: number;
  side: "left" | "right";
  onRecorded: (uri: string) => void;
  onCancel: () => void;
};

export default function RecordScreen({ durationSec, side, onRecorded, onCancel }: Props) {
  const cameraRef = useRef<CameraView>(null);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micPermission, requestMicPermission] = useMicrophonePermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(durationSec);

  useEffect(() => {
    if (Platform.OS !== "web") {
      if (!cameraPermission?.granted) requestCameraPermission();
      if (!micPermission?.granted) requestMicPermission();
    }
  }, [Platform.OS, cameraPermission?.granted, micPermission?.granted]);

  useEffect(() => {
    if (!isRecording) return;
    if (secondsLeft <= 0) return;
    const timer = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(timer);
  }, [isRecording, secondsLeft]);

  const startRecording = async () => {
    if (!cameraRef.current) return;
    setIsRecording(true);
    setSecondsLeft(durationSec);
    try {
      const video = await cameraRef.current.recordAsync({ maxDuration: durationSec });
      if (video?.uri) onRecorded(video.uri);
    } finally {
      setIsRecording(false);
    }
  };

  const uploadVideoFile = () => {
    if (typeof document === "undefined") return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const uri = URL.createObjectURL(file);
      onRecorded(uri);
    };
    input.click();
  };

  if (Platform.OS === "web") {
    return (
      <View style={styles.container}>
        <View style={styles.overlay}>
          <Text style={styles.instructions}>
            Testing {side} side. Choose how you want to provide the video for this prototype.
          </Text>
        </View>

        <View style={styles.webActions}>
          <TouchableOpacity style={styles.secondaryButtonWide} onPress={onCancel}>
            <Text style={styles.secondaryButtonText}>Back</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.recordButtonWide} onPress={uploadVideoFile}>
            <Text style={styles.primaryButtonText}>Upload Video</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!cameraPermission || !micPermission) {
    return <View style={styles.container} />;
  }

  if (!cameraPermission.granted || !micPermission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Camera and microphone access are required to record the head impulse test.</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={() => { requestCameraPermission(); requestMicPermission(); }}>
          <Text style={styles.primaryButtonText}>Grant Permissions</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="front" mode="video">
        <View style={styles.overlay}>
          <Text style={styles.instructions}>
            Testing {side} side. Hold the phone steady at arm's length, look at the camera, then
            perform a brief, small head turn.
          </Text>
          {isRecording && <Text style={styles.timer}>{secondsLeft}s</Text>}
        </View>
      </CameraView>

      <View style={styles.controls}>
        <TouchableOpacity style={styles.secondaryButton} onPress={onCancel} disabled={isRecording}>
          <Text style={styles.secondaryButtonText}>Back</Text>
        </TouchableOpacity>

        {!isRecording ? (
          <View style={styles.nativeActions}>
            <TouchableOpacity style={styles.recordButton} onPress={startRecording}>
              <Text style={styles.primaryButtonText}>Record Video</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.uploadButton} onPress={uploadVideoFile}>
              <Text style={styles.primaryButtonText}>Upload Video</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <TouchableOpacity
            style={styles.stopButton}
            onPress={() => cameraRef.current?.stopRecording()}
          >
            <Text style={styles.primaryButtonText}>Stop</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
  overlay: {
    flex: 1,
    justifyContent: "space-between",
    padding: 20,
    paddingTop: 60,
  },
  instructions: {
    color: "#fff",
    fontSize: 16,
    textAlign: "center",
    backgroundColor: "rgba(0,0,0,0.5)",
    padding: 12,
    borderRadius: 10,
  },
  timer: {
    alignSelf: "center",
    color: "#fff",
    fontSize: 48,
    fontWeight: "700",
    marginBottom: 40,
  },
  controls: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: 20,
    backgroundColor: "#000",
  },
  nativeActions: {
    flexDirection: "row",
    gap: 12,
    alignItems: "center",
  },
  webActions: {
    gap: 12,
    padding: 20,
    backgroundColor: "#000",
  },
  recordButton: {
    backgroundColor: "#dc2626",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
  },
  recordButtonWide: {
    backgroundColor: "#dc2626",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
    alignItems: "center",
  },
  uploadButton: {
    backgroundColor: "#2563eb",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
  },
  stopButton: {
    backgroundColor: "#374151",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
  },
  secondaryButton: { paddingVertical: 14, paddingHorizontal: 16 },
  secondaryButtonWide: {
    backgroundColor: "#1f2937",
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
    alignItems: "center",
  },
  secondaryButtonText: { color: "#9ca3af", fontSize: 16 },
  primaryButton: {
    backgroundColor: "#2563eb",
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 20,
    paddingHorizontal: 24,
  },
  primaryButtonText: { color: "#fff", fontSize: 16, fontWeight: "700" },
  message: { color: "#fff", fontSize: 16, textAlign: "center", margin: 20 },
});
