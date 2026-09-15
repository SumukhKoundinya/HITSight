import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import type { ScreeningResult } from "./api";
import type { FacilityType } from "./CareFinderScreen";
import { colors, radius, shadow, spacing } from "./theme";

type Props = {
  result: ScreeningResult;
  onRestart: () => void;
  onFindCare: (facility: FacilityType) => void;
};

export function recommendedFacility(result: ScreeningResult): FacilityType {
  return result.stroke_risk_flag || result.hit_predicted_class === "Abnormal" ? "er" : "neurologist";
}

export default function ResultScreen({ result, onRestart, onFindCare }: Props) {
  const riskPercent = Math.round(result.stroke_risk_probability * 100);
  const facility = recommendedFacility(result);

  return (
    <ScrollView style={styles.flex} contentContainerStyle={styles.container}>
      <Text style={styles.title}>Screening Result</Text>

      <View style={[styles.card, styles.riskCard, result.stroke_risk_flag ? styles.cardWarning : styles.cardOk]}>
        <Text style={styles.cardLabel}>Fused Stroke-Risk Probability</Text>
        <Text style={styles.riskValue}>{riskPercent}%</Text>
        <Text style={styles.cardLabel}>
          {result.stroke_risk_flag ? "Flagged — recommend clinical follow-up" : "Not flagged"}
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Head Impulse Test Classification</Text>
        <Text style={styles.hitClass}>{result.hit_predicted_class}</Text>

        {Object.entries(result.hit_class_probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, prob]) => (
            <View key={cls} style={styles.barRow}>
              <Text style={styles.barLabel}>{cls}</Text>
              <View style={styles.barTrack}>
                <View style={[styles.barFill, { width: `${Math.round(prob * 100)}%` }]} />
              </View>
              <Text style={styles.barValue}>{Math.round(prob * 100)}%</Text>
            </View>
          ))}
      </View>

      <TouchableOpacity style={styles.primaryButton} onPress={() => onFindCare(facility)}>
        <Text style={styles.primaryButtonText}>
          Find Nearest {facility === "er" ? "Emergency Room" : "Neurologist"}
        </Text>
      </TouchableOpacity>

      <Text style={styles.disclaimer}>
        This is a screening prototype, not a diagnostic device. Consult a clinician for any
        concerning symptoms.
      </Text>

      <TouchableOpacity style={styles.secondaryButton} onPress={onRestart}>
        <Text style={styles.secondaryButtonText}>Start New Screening</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  container: { padding: spacing.lg, paddingTop: 60, paddingBottom: spacing.xl },
  title: { fontSize: 24, fontWeight: "800", color: colors.textPrimary, marginBottom: spacing.lg },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    ...shadow,
  },
  riskCard: { alignItems: "center" },
  cardOk: { backgroundColor: colors.successBg },
  cardWarning: { backgroundColor: colors.dangerBg },
  cardLabel: { fontSize: 14, color: colors.textSecondary, marginTop: 4 },
  riskValue: { fontSize: 42, fontWeight: "800", color: colors.textPrimary },
  sectionTitle: { fontSize: 15, fontWeight: "700", color: colors.textPrimary, marginBottom: spacing.sm },
  hitClass: { fontSize: 20, fontWeight: "700", color: colors.primary, marginBottom: spacing.md },
  barRow: { flexDirection: "row", alignItems: "center", marginBottom: spacing.sm },
  barLabel: { width: 130, fontSize: 12, color: colors.textSecondary },
  barTrack: { flex: 1, height: 10, backgroundColor: colors.border, borderRadius: 5, overflow: "hidden" },
  barFill: { height: "100%", backgroundColor: colors.primary },
  barValue: { width: 40, textAlign: "right", fontSize: 12, color: colors.textSecondary },
  disclaimer: { marginTop: spacing.lg, fontSize: 12, color: colors.textMuted, textAlign: "center" },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: spacing.sm,
  },
  primaryButtonText: { color: "#fff", fontSize: 16, fontWeight: "700" },
  secondaryButton: {
    borderColor: colors.primary,
    borderWidth: 1,
    borderRadius: radius.md,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: spacing.md,
  },
  secondaryButtonText: { color: colors.primary, fontSize: 16, fontWeight: "700" },
});

