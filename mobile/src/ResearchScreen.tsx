import { Linking, ScrollView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { colors, radius, shadow, spacing } from "./theme";

type Citation = {
  title: string;
  authors: string;
  journal: string;
  year: number;
  summary: string;
  doi: string;
};

// Peer-reviewed sources behind the HINTS (Head-Impulse, Nystagmus, Test-of-Skew)
// exam this app automates the head-impulse component of. Verified against
// Crossref metadata — do not add citations without verifying the DOI first.
const CITATIONS: Citation[] = [
  {
    title: "HINTS to Diagnose Stroke in the Acute Vestibular Syndrome",
    authors: "Kattah JC, Talkad AV, Wang DZ, Hsieh YH, Newman-Toker DE",
    journal: "Stroke",
    year: 2009,
    summary:
      "The original prospective study defining the HINTS exam. An abnormal HINTS battery (normal head impulse test, direction-changing nystagmus, or skew deviation) was 100% sensitive and 96% specific for stroke in acute vestibular syndrome — more sensitive than early MRI.",
    doi: "10.1161/STROKEAHA.109.551234",
  },
  {
    title: "Normal Head Impulse Test Differentiates Acute Cerebellar Strokes from Vestibular Neuritis",
    authors: "Newman-Toker DE, Kattah JC, Alvernia JE, Wang DZ",
    journal: "Neurology",
    year: 2008,
    summary:
      "Shows that a normal (rather than abnormal) head impulse test in a patient with acute vertigo is itself a red flag for a central cause such as cerebellar stroke — the core logic this app's CNN classifier is trained to detect.",
    doi: "10.1212/01.wnl.0000314685.01433.0d",
  },
  {
    title: "Diagnosis and Initial Management of Cerebellar Infarction",
    authors: "Edlow JA, Newman-Toker DE, Savitz SI",
    journal: "The Lancet Neurology",
    year: 2008,
    summary:
      "Review of why cerebellar/posterior-circulation strokes are so frequently missed in emergency settings, motivating bedside oculomotor screening tools like HINTS.",
    doi: "10.1016/S1474-4422(08)70216-3",
  },
];

type Props = {
  onBack: () => void;
};

export default function ResearchScreen({ onBack }: Props) {
  return (
    <ScrollView style={styles.flex} contentContainerStyle={styles.container}>
      <TouchableOpacity onPress={onBack} style={styles.backRow}>
        <Text style={styles.backText}>‹ Back</Text>
      </TouchableOpacity>

      <Text style={styles.title}>The Research Behind HINTSight</Text>
      <Text style={styles.intro}>
        HINTSight automates the head-impulse component of HINTS — a bedside oculomotor exam
        clinicians already use to tell a benign inner-ear problem from a dangerous posterior-circulation
        stroke in patients with acute vertigo. Here is the peer-reviewed evidence it's based on.
      </Text>

      {CITATIONS.map((c) => (
        <View key={c.doi} style={styles.card}>
          <Text style={styles.cardTitle}>{c.title}</Text>
          <Text style={styles.cardMeta}>
            {c.authors} · {c.journal} ({c.year})
          </Text>
          <Text style={styles.cardSummary}>{c.summary}</Text>
          <TouchableOpacity onPress={() => Linking.openURL(`https://doi.org/${c.doi}`)}>
            <Text style={styles.link}>doi.org/{c.doi}</Text>
          </TouchableOpacity>
        </View>
      ))}

      <Text style={styles.disclaimer}>
        HINTSight is a research/screening prototype inspired by this literature. It has not been
        clinically validated and is not a substitute for an in-person exam by a clinician.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  container: { padding: spacing.lg, paddingTop: 60, paddingBottom: spacing.xl },
  backRow: { marginBottom: spacing.md },
  backText: { color: colors.primary, fontSize: 16, fontWeight: "600" },
  title: { fontSize: 24, fontWeight: "800", color: colors.textPrimary, marginBottom: spacing.sm },
  intro: { fontSize: 14, color: colors.textSecondary, lineHeight: 20, marginBottom: spacing.lg },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    ...shadow,
  },
  cardTitle: { fontSize: 15, fontWeight: "700", color: colors.textPrimary, marginBottom: 4 },
  cardMeta: { fontSize: 12, color: colors.textMuted, marginBottom: spacing.sm },
  cardSummary: { fontSize: 13, color: colors.textSecondary, lineHeight: 19, marginBottom: spacing.sm },
  link: { fontSize: 13, color: colors.primary, fontWeight: "600" },
  disclaimer: { fontSize: 12, color: colors.textMuted, textAlign: "center", marginTop: spacing.md },
});
