import { useState } from "react";
import {
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { submitScreening, type QuizAnswers, type ScreeningResult } from "./src/api";
import CareFinderScreen, { type FacilityType } from "./src/CareFinderScreen";
import QuizScreen from "./src/QuizScreen";
import RecordScreen from "./src/RecordScreen";
import ResearchScreen from "./src/ResearchScreen";
import ResultScreen from "./src/ResultScreen";
import { colors } from "./src/theme";

type Stage = "quiz" | "record" | "submitting" | "result" | "error" | "research" | "care";

const RECORDING_DURATION_SEC = 10;
const HIT_SIDE = "left" as const;

export default function App() {
  const [stage, setStage] = useState<Stage>("quiz");
  const [previousStage, setPreviousStage] = useState<Stage>("quiz");
  const [quizAnswers, setQuizAnswers] = useState<QuizAnswers | null>(null);
  const [result, setResult] = useState<ScreeningResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [careFacility, setCareFacility] = useState<FacilityType>("er");

  const handleQuizSubmit = (answers: QuizAnswers) => {
    setQuizAnswers(answers);
    setStage("record");
  };

  const handleRecorded = async (videoUri: string) => {
    if (!quizAnswers) return;
    setStage("submitting");
    try {
      const screeningResult = await submitScreening(
        videoUri,
        quizAnswers,
        HIT_SIDE,
        RECORDING_DURATION_SEC
      );
      setResult(screeningResult);
      setStage("result");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setStage("error");
    }
  };

  const restart = () => {
    setQuizAnswers(null);
    setResult(null);
    setErrorMessage(null);
    setStage("quiz");
  };

  const openResearch = (from: Stage) => {
    setPreviousStage(from);
    setStage("research");
  };

  const openCareFinder = (facility: FacilityType) => {
    setCareFacility(facility);
    setPreviousStage("result");
    setStage("care");
  };

  return (
    <SafeAreaView style={styles.flex}>
      <StatusBar barStyle="dark-content" />

      {stage === "quiz" && <QuizScreen onSubmit={handleQuizSubmit} onOpenResearch={() => openResearch("quiz")} />}

      {stage === "record" && (
        <RecordScreen
          durationSec={RECORDING_DURATION_SEC}
          side={HIT_SIDE}
          onRecorded={handleRecorded}
          onCancel={() => setStage("quiz")}
        />
      )}

      {stage === "submitting" && (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loadingText}>Analyzing recording and risk factors…</Text>
        </View>
      )}

      {stage === "result" && result && (
        <ResultScreen result={result} onRestart={restart} onFindCare={openCareFinder} />
      )}

      {stage === "research" && <ResearchScreen onBack={() => setStage(previousStage)} />}

      {stage === "care" && (
        <CareFinderScreen facility={careFacility} onBack={() => setStage(previousStage)} />
      )}

      {stage === "error" && (
        <View style={styles.centered}>
          <Text style={styles.errorTitle}>Something went wrong</Text>
          <Text style={styles.errorMessage}>{errorMessage}</Text>
          <Text style={styles.errorHint} onPress={restart}>
            Tap to try again
          </Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: "#fff" },
  centered: { flex: 1, justifyContent: "center", alignItems: "center", padding: 20 },
  loadingText: { marginTop: 16, fontSize: 16, color: "#374151" },
  errorTitle: { fontSize: 20, fontWeight: "700", color: "#dc2626", marginBottom: 12 },
  errorMessage: { fontSize: 14, color: "#374151", textAlign: "center", marginBottom: 20 },
  errorHint: { fontSize: 16, color: "#2563eb", fontWeight: "600" },
});
