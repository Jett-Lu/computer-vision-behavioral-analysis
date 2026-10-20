from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import (
    DetectedFace,
    IdentityConsensus,
    IdentityMatch,
    OpenCVFaceIdentifier,
    face_box,
    offset_box,
)


class IdentityHelpersTest(unittest.TestCase):
    def test_face_box_converts_yunet_width_height_to_corners(self) -> None:
        raw_face = np.array([10.2, 20.4, 30.3, 40.6, *([0.0] * 10), 0.95])

        self.assertEqual(face_box(raw_face), (10, 20, 40, 61))

    def test_face_box_is_clamped_to_the_image(self) -> None:
        raw_face = np.array([-10.0, 20.0, 150.0, 100.0, *([0.0] * 10), 0.95])

        self.assertEqual(
            face_box(raw_face, image_width=100, image_height=80),
            (0, 20, 100, 80),
        )

    def test_offset_box_moves_crop_box_to_frame_coordinates(self) -> None:
        self.assertEqual(offset_box((5, 6, 30, 40), 100, 200), (105, 206, 130, 240))

    def test_face_quality_rejects_small_or_low_confidence_faces(self) -> None:
        identifier = OpenCVFaceIdentifier.__new__(OpenCVFaceIdentifier)
        identifier.min_face_size = 64
        identifier.min_face_confidence = 0.85

        small = DetectedFace(np.zeros(15), (0, 0, 40, 40), 0.95)
        uncertain = DetectedFace(np.zeros(15), (0, 0, 100, 100), 0.70)
        usable = DetectedFace(np.zeros(15), (0, 0, 100, 100), 0.95)

        self.assertFalse(identifier.is_face_usable(small))
        self.assertFalse(identifier.is_face_usable(uncertain))
        self.assertTrue(identifier.is_face_usable(usable))


class IdentityConsensusTest(unittest.TestCase):
    def test_identity_requires_repeated_agreement(self) -> None:
        consensus = IdentityConsensus(window_size=5, required_matches=3, ttl_frames=10)
        match = IdentityMatch("Alex", 0.72, True, 0.40, 0.32)

        self.assertFalse(consensus.observe(match, 0).confirmed)
        self.assertFalse(consensus.observe(match, 1).confirmed)
        decision = consensus.observe(match, 2)

        self.assertTrue(decision.confirmed)
        self.assertEqual(decision.name, "Alex")
        self.assertAlmostEqual(decision.score, 0.72)

    def test_confirmed_identity_expires(self) -> None:
        consensus = IdentityConsensus(window_size=3, required_matches=2, ttl_frames=5)
        match = IdentityMatch("Alex", 0.70, True)
        consensus.observe(match, 0)
        consensus.observe(match, 1)

        self.assertTrue(consensus.current(5).confirmed)
        self.assertFalse(consensus.current(7).confirmed)

    def test_conflicting_match_hides_cached_identity_until_reconfirmed(self) -> None:
        consensus = IdentityConsensus(window_size=3, required_matches=2, ttl_frames=20)
        alex = IdentityMatch("Alex", 0.70, True)
        jordan = IdentityMatch("Jordan", 0.72, True)
        consensus.observe(alex, 0)
        consensus.observe(alex, 1)

        conflicted = consensus.observe(jordan, 2)

        self.assertFalse(conflicted.confirmed)
        self.assertEqual(conflicted.name, "Unknown")

    def test_failed_observations_do_not_extend_identity_expiry(self) -> None:
        for failed_match in (None,):
            with self.subTest(failed_match=failed_match):
                consensus = IdentityConsensus(ttl_frames=5)
                match = IdentityMatch("Alex", 0.70, True)
                for frame in (0, 1, 2):
                    consensus.observe(match, frame)

                self.assertTrue(consensus.observe(failed_match, 3).confirmed)
                self.assertTrue(consensus.current(7).confirmed)
                self.assertFalse(consensus.current(8).confirmed)

    def test_failed_observation_after_expiry_does_not_revive_identity(self) -> None:
        consensus = IdentityConsensus(ttl_frames=5)
        match = IdentityMatch("Alex", 0.70, True)
        for frame in (0, 1, 2):
            consensus.observe(match, frame)

        decision = consensus.observe(None, 10)

        self.assertFalse(decision.confirmed)
        self.assertEqual(decision.name, "Unknown")
        self.assertIsNone(decision.score)

    def test_match_after_expiry_requires_fresh_consensus(self) -> None:
        consensus = IdentityConsensus(ttl_frames=5)
        match = IdentityMatch("Alex", 0.70, True)
        for frame in (0, 1, 2):
            consensus.observe(match, frame)

        self.assertFalse(consensus.observe(match, 10).confirmed)
        self.assertFalse(consensus.observe(match, 11).confirmed)
        self.assertTrue(consensus.observe(match, 12).confirmed)

    def test_fresh_matching_observation_renews_unexpired_identity(self) -> None:
        consensus = IdentityConsensus(ttl_frames=5)
        match = IdentityMatch("Alex", 0.70, True)
        for frame in (0, 1, 2):
            consensus.observe(match, frame)

        self.assertTrue(consensus.observe(match, 7).confirmed)
        self.assertTrue(consensus.current(12).confirmed)
        self.assertFalse(consensus.current(13).confirmed)


if __name__ == "__main__":
    unittest.main()
