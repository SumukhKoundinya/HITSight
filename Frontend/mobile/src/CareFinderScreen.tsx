import * as Location from "expo-location";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Linking,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { API_BASE_URL } from "./api";
import { colors, radius, shadow, spacing } from "./theme";

export type FacilityType = "er" | "neurologist";

type Place = {
  name: string;
  address: string | null;
  lat: number;
  lng: number;
  place_id: string;
  rating: number | null;
  open_now: boolean | null;
  distance_m: number;
  maps_url: string;
};

type Props = {
  facility: FacilityType;
  onBack: () => void;
};

const FACILITY_LABEL: Record<FacilityType, string> = {
  er: "Emergency Room",
  neurologist: "Neurologist",
};

function openFallbackSearch(facility: FacilityType) {
  const query = facility === "er" ? "emergency room near me" : "neurologist near me";
  Linking.openURL(`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`);
}

export default function CareFinderScreen({ facility, onBack }: Props) {
  const [status, setStatus] = useState<"loading" | "ready" | "error" | "no-key">("loading");
  const [places, setPlaces] = useState<Place[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const { status: permStatus } = await Location.requestForegroundPermissionsAsync();
        if (permStatus !== "granted") {
          throw new Error("Location permission was denied.");
        }

        const position = await Location.getCurrentPositionAsync({});
        const { latitude, longitude } = position.coords;

        const response = await fetch(
          `${API_BASE_URL}/nearby-care?lat=${latitude}&lng=${longitude}&facility=${facility}`
        );

        if (response.status === 503) {
          if (!cancelled) setStatus("no-key");
          return;
        }
        if (!response.ok) {
          throw new Error(`Server error (${response.status})`);
        }

        const data = await response.json();
        if (!cancelled) {
          setPlaces(data.results ?? []);
          setStatus("ready");
        }
      } catch (err) {
        if (!cancelled) {
          setErrorMessage(err instanceof Error ? err.message : String(err));
          setStatus("error");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [facility]);

  return (
    <ScrollView style={styles.flex} contentContainerStyle={styles.container}>
      <TouchableOpacity onPress={onBack} style={styles.backRow}>
        <Text style={styles.backText}>‹ Back</Text>
      </TouchableOpacity>

      <Text style={styles.title}>Nearest {FACILITY_LABEL[facility]}</Text>

      {status === "loading" && (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loadingText}>Finding nearby care…</Text>
        </View>
      )}

      {(status === "error" || status === "no-key") && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            {status === "no-key" ? "Location search isn't configured yet" : "Couldn't fetch nearby locations"}
          </Text>
          {errorMessage && <Text style={styles.cardMeta}>{errorMessage}</Text>}
          <TouchableOpacity style={styles.primaryButton} onPress={() => openFallbackSearch(facility)}>
            <Text style={styles.primaryButtonText}>Open Google Maps Search</Text>
          </TouchableOpacity>
        </View>
      )}

      {status === "ready" && places.length === 0 && (
        <Text style={styles.cardMeta}>No results found nearby.</Text>
      )}

      {status === "ready" &&
        places.map((place) => (
          <View key={place.place_id} style={styles.card}>
            <Text style={styles.cardTitle}>{place.name}</Text>
            {place.address && <Text style={styles.cardMeta}>{place.address}</Text>}
            <Text style={styles.cardMeta}>
              {(place.distance_m / 1000).toFixed(1)} km away
              {place.rating ? ` · ${place.rating}★` : ""}
              {place.open_now === true ? " · Open now" : place.open_now === false ? " · Closed" : ""}
            </Text>
            <TouchableOpacity style={styles.secondaryButton} onPress={() => Linking.openURL(place.maps_url)}>
              <Text style={styles.secondaryButtonText}>Open in Maps</Text>
            </TouchableOpacity>
          </View>
        ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  container: { padding: spacing.lg, paddingTop: 60, paddingBottom: spacing.xl },
  backRow: { marginBottom: spacing.md },
  backText: { color: colors.primary, fontSize: 16, fontWeight: "600" },
  title: { fontSize: 22, fontWeight: "800", color: colors.textPrimary, marginBottom: spacing.lg },
  centered: { alignItems: "center", paddingVertical: spacing.xl },
  loadingText: { marginTop: spacing.md, color: colors.textSecondary },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    ...shadow,
  },
  cardTitle: { fontSize: 15, fontWeight: "700", color: colors.textPrimary, marginBottom: 4 },
  cardMeta: { fontSize: 13, color: colors.textSecondary, marginBottom: spacing.xs },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: radius.md,
    paddingVertical: 12,
    alignItems: "center",
    marginTop: spacing.sm,
  },
  primaryButtonText: { color: "#fff", fontSize: 15, fontWeight: "700" },
  secondaryButton: {
    borderColor: colors.primary,
    borderWidth: 1,
    borderRadius: radius.md,
    paddingVertical: 10,
    alignItems: "center",
    marginTop: spacing.sm,
  },
  secondaryButtonText: { color: colors.primary, fontSize: 14, fontWeight: "700" },
});
