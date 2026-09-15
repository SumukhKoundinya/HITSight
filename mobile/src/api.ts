/**
 * Point this to the LAN IP of the machine running the FastAPI backend.
 * Use the same machine that is hosting the Python API on port 8000.
 */
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? "http://172.20.10.4:8000";

export type QuizAnswers = {
  gender: "male" | "female";
  age: number;
  hypertension: boolean;
  heart_disease: boolean;
  ever_married: boolean;
  work_type: "Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked";
  Residence_type: "Urban" | "Rural";
  avg_glucose_level: number;
  bmi: number;
  smoking_status: "never smoked" | "formerly smoked" | "smokes" | "Unknown";
};

export type ScreeningResult = {
  hit_predicted_class: string;
  hit_class_probabilities: Record<string, number>;
  stroke_risk_probability: number;
  stroke_risk_flag: boolean;
};

async function buildVideoFormData(videoUri: string, formData: FormData): Promise<void> {
  if (typeof window !== "undefined" && videoUri.startsWith("blob:")) {
    const blob = await fetch(videoUri).then((res) => res.blob());
    const file = new File([blob], "hit_recording.mp4", { type: blob.type || "video/mp4" });
    formData.append("video", file);
    return;
  }

  // React Native native-file upload path.
  formData.append("video", {
    uri: videoUri,
    name: "hit_recording.mp4",
    type: "video/mp4",
  } as unknown as Blob);
}

export async function submitScreening(
  videoUri: string,
  quiz: QuizAnswers,
  side: "left" | "right" = "left",
  durationSec = 10
): Promise<ScreeningResult> {
  const formData = new FormData();
  await buildVideoFormData(videoUri, formData);
  formData.append("quiz", JSON.stringify(quiz));
  formData.append("side", side);
  formData.append("duration_sec", String(durationSec));

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Server error (${response.status}): ${detail}`);
  }

  return (await response.json()) as ScreeningResult;
}
