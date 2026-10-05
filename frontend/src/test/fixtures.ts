/* A complete, small Result for tests. Every field of the real contract is present, no casts. */
import type { AttributionTrack, Result } from "../services/demoApi";

const track = (
  method: AttributionTrack["method"],
  values: number[],
  signed: boolean,
): AttributionTrack => ({
  method,
  signed,
  target: "CYT",
  values,
  window: signed ? 5 : null,
  stride: signed ? 1 : null,
  forward_passes: signed ? 8 : 1,
  scored_length: values.length,
  claim: `claim for ${method}`,
});

const OCCLUSION = [0.1, -0.2, 0.3, 0, 0.05, -0.4, 0.2, 0.1, 0, 0.3, -0.1, 0.2];

export function makeResult(over: Partial<Result> = {}): Result {
  const base: Result = {
    id: "TEST1",
    length: 12,
    sequence: "MKTAYIAKQRQI",
    sequence_sha256: "sha-test",
    organism_group: "eukaryote",
    localization: [
      {
        code: "CYT",
        name: "Cytoplasm",
        class_name: "cytoplasm",
        probability: 0.8,
        raw: 1.2,
        predicted: true,
        source: "model",
        threshold: 0.5,
        calibrated: true,
        calibration_method: "isotonic",
        confidence_band: "high",
      },
    ],
    top_location: "CYT",
    abstained: false,
    no_confident_call: false,
    calibration_version: "0.10.0",
    threshold_policy: "per_label_mcc",
    task_sources: {
      localization: {
        source: "model",
        calibrated: true,
        bundle_version: "0.10.0",
        bundle_sha256: "a".repeat(64),
        calibration_version: "0.10.0",
      },
      membrane_type: {
        source: "motif",
        calibrated: false,
        bundle_version: null,
        bundle_sha256: null,
        calibration_version: null,
      },
      signal_peptide: {
        source: "model",
        calibrated: true,
        bundle_version: "0.11.0",
        bundle_sha256: "b".repeat(64),
        calibration_version: "0.11.0",
      },
      go_function: {
        source: "model",
        calibrated: true,
        bundle_version: "0.12.0",
        bundle_sha256: "c".repeat(64),
        calibration_version: "0.12.0",
      },
    },
    tau: null,
    temperature: null,
    membrane_type: { soluble: 0.9 },
    sorting_signal: {
      type: "signal_peptide",
      probability: 0.9,
      region: [0, 5],
      cleavage_site: 5,
      candidates: { signal_peptide: 0.9 },
    },
    go_function: [
      {
        go_id: "GO:0003824",
        name: "catalytic activity",
        aspect: "molecular_function",
        probability: 0.7,
        raw: 0.9,
        predicted: true,
        threshold: 0.4,
        source: "model",
        calibrated: true,
      },
    ],
    features: { tm_segments: [[8, 11]] },
    hydropathy: [1.8, -3.9, -0.7, 1.8, -1.3, 4.5, 1.8, -3.9, -3.5, -4.5, -3.5, 4.5],
    attribution: OCCLUSION,
    attribution_source: "model",
    attribution_method: "occlusion",
    attribution_signed: true,
    attribution_target: "CYT",
    attribution_tracks: {
      occlusion: track("occlusion", OCCLUSION, true),
      pooling_attention: track(
        "pooling_attention",
        [0.1, 0.2, 0.1, 0, 0, 0.3, 0.1, 0, 0, 0.1, 0, 0.1],
        false,
      ),
    },
    signal_track: track("crf_marginal", [0.9, 0.9, 0.8, 0.7, 0.6, 0.1, 0, 0, 0, 0, 0, 0], false),
    consistency_flags: [],
    evidence: [],
    evidence_go: [],
    evidence_status: {
      available: true,
      unavailable_reason: null,
      swissprot_release: "2026_03",
      bundle_version: "0.13.0",
      bundle_sha256: "d".repeat(64),
      source: "homology",
    },
    evidence_agreement: null,
    provenance: {
      model_version: "1.0.0-test",
      config_hash: "cfg-test",
      rules_version: "demo-0.1.0",
      experiment_id: "EXP-TEST",
      predictor: "test",
    },
  };
  return { ...base, ...over };
}
