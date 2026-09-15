import { useState } from "react";
import {
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import type { QuizAnswers } from "./api";
import { colors, radius, shadow, spacing } from "./theme";

type Props = {
  onSubmit: (answers: QuizAnswers) => void;
  onOpenResearch: () => void;
};

export default function QuizScreen({ onSubmit, onOpenResearch }: Props) {
  const [gender, setGender] = useState<"male" | "female">("female");
  const [age, setAge] = useState("");
  const [hypertension, setHypertension] = useState(false);
  const [heartDisease, setHeartDisease] = useState(false);
  const [everMarried, setEverMarried] = useState(false);
  const [workType, setWorkType] = useState<"Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked">("Private");
  const [residenceType, setResidenceType] = useState<"Urban" | "Rural">("Urban");
  const [smokingStatus, setSmokingStatus] = useState<"never smoked" | "formerly smoked" | "smokes" | "Unknown">("Unknown");
  const [glucose, setGlucose] = useState("");
  const [bmi, setBmi] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleContinue = () => {
    const ageNum = parseFloat(age);
    const glucoseNum = parseFloat(glucose);
    const bmiNum = parseFloat(bmi);

    if (!Number.isFinite(ageNum) || !Number.isFinite(glucoseNum) || !Number.isFinite(bmiNum)) {
      setError("Please fill in age, glucose level, and BMI with valid numbers.");
      return;
    }

    setError(null);
    onSubmit({
      gender,
      age: ageNum,
      hypertension,
      heart_disease: heartDisease,
      ever_married: everMarried,
      work_type: workType,
      Residence_type: residenceType,
      avg_glucose_level: glucoseNum,
      bmi: bmiNum,
      smoking_status: smokingStatus,
    });
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <View>
            <Text style={styles.brand}>HINTSight</Text>
            <Text style={styles.title}>Risk Factor Questionnaire</Text>
          </View>
          <TouchableOpacity onPress={onOpenResearch} style={styles.researchLink}>
            <Text style={styles.researchLinkText}>ℹ Research</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.label}>Gender</Text>
        <View style={styles.row}>
          {(["female", "male"] as const).map((option) => (
            <TouchableOpacity
              key={option}
              style={[styles.pill, gender === option && styles.pillActive]}
              onPress={() => setGender(option)}
            >
              <Text style={gender === option ? styles.pillTextActive : styles.pillText}>
                {option === "female" ? "Female" : "Male"}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <Text style={styles.label}>Age</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          value={age}
          onChangeText={setAge}
          placeholder="e.g. 62"
        />

        <Text style={styles.label}>Average glucose level (mg/dL)</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          value={glucose}
          onChangeText={setGlucose}
          placeholder="e.g. 105"
        />

        <Text style={styles.label}>BMI</Text>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          value={bmi}
          onChangeText={setBmi}
          placeholder="e.g. 27.5"
        />

        <SwitchRow label="Hypertension" value={hypertension} onChange={setHypertension} />
        <SwitchRow label="Heart disease" value={heartDisease} onChange={setHeartDisease} />
        <SwitchRow label="Ever married" value={everMarried} onChange={setEverMarried} />
        <ChoiceRow label="Work type" value={workType} options={["Private", "Self-employed", "Govt_job", "children", "Never_worked"]} onChange={setWorkType} />
        <ChoiceRow label="Residence type" value={residenceType} options={["Urban", "Rural"]} onChange={setResidenceType} />
        <ChoiceRow label="Smoking status" value={smokingStatus} options={["never smoked", "formerly smoked", "smokes", "Unknown"]} onChange={setSmokingStatus} />

        {error && <Text style={styles.error}>{error}</Text>}

        <TouchableOpacity style={styles.primaryButton} onPress={handleContinue}>
          <Text style={styles.primaryButtonText}>Continue to Head Impulse Test</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

function SwitchRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <View style={styles.switchRow}>
      <Text style={styles.switchLabel}>{label}</Text>
      <Switch value={value} onValueChange={onChange} />
    </View>
  );
}

function ChoiceRow<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
}) {
  return (
    <View>
      <Text style={styles.label}>{label}</Text>
      <View style={styles.row}>
        {options.map((option) => (
          <TouchableOpacity
            key={option}
            style={[styles.pill, value === option && styles.pillActive]}
            onPress={() => onChange(option)}
          >
            <Text style={value === option ? styles.pillTextActive : styles.pillText}>{option}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  container: { padding: spacing.lg, paddingTop: 60, paddingBottom: spacing.xl },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: spacing.lg,
  },
  brand: { fontSize: 13, fontWeight: "700", color: colors.primary, letterSpacing: 1, marginBottom: 2 },
  title: { fontSize: 22, fontWeight: "800", color: colors.textPrimary },
  researchLink: { paddingVertical: 6, paddingHorizontal: 10 },
  researchLinkText: { color: colors.primary, fontSize: 13, fontWeight: "700" },
  label: { fontSize: 14, fontWeight: "600", marginTop: spacing.md, marginBottom: 6, color: colors.textSecondary },
  input: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.sm,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    backgroundColor: colors.card,
  },
  row: { flexDirection: "row", gap: 10 },
  pill: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
  },
  pillActive: { backgroundColor: colors.primary, borderColor: colors.primary },
  pillText: { color: colors.textSecondary },
  pillTextActive: { color: "#fff", fontWeight: "600" },
  switchRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: spacing.md,
    backgroundColor: colors.card,
    borderRadius: radius.md,
    padding: spacing.sm,
    ...shadow,
  },
  switchLabel: { flex: 1, marginRight: 12, fontSize: 14, color: colors.textSecondary },
  error: { color: colors.danger, marginTop: spacing.md },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: spacing.xl,
  },
  primaryButtonText: { color: "#fff", fontSize: 16, fontWeight: "700" },
});
