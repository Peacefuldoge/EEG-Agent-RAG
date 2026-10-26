# EEG Method Notes (synthetic demonstration data)

These notes are synthetic examples designed for validating a retrieval pipeline. They are not a clinical protocol, a peer-reviewed paper, or evidence for a specific patient.

## EEG preprocessing

An EEG preprocessing report should identify the sampling frequency, acquisition configuration, reference scheme, and any bandpass and notch filters. The choice of low and high cutoff frequencies should be justified for the measurement goal rather than copied from an unrelated study. Operators should record filter type, order or transition band, software version, and whether processing was offline or online. If channels are removed or interpolated, retain a machine-readable list and the decision criteria.

When using ICA or another artifact-correction algorithm, record the input signal preparation, number of fitted components, component selection rationale, and whether all subjects followed the same procedure. When original measurements are unavailable, a generated answer must acknowledge that it cannot verify preprocessing decisions. This synthetic passage is only a documentation example and must not be presented as proof that a particular numerical filter is optimal for EEG depression detection.

## Quality control

Quality-control checks may include detecting flat channels, abnormal signal amplitudes, power-line interference, and discontinuities. A workflow can flag suspicious segments for manual review without automatically diagnosing a patient. A QC report should link each warning to the input file, recording or sensor and the processing steps that produced it. Bad-channel interpolation must be explicitly labelled because an interpolated channel is not an independent measurement.

For repeatability, record the acquisition file hash, selected channels, analysis time window, threshold configuration, preprocessing warnings, and the software environment. An automated system should not hide missing electrodes or silently substitute channel names. Error states should be visible and an operator should be able to re-run failed recordings without duplicating previously completed results. These notes are synthetic and are intended to test the metadata fields of the RAG system.

## Retrieval-augmented scientific reports

A retrieval-augmented scientific assistant should connect method-related claims to traceable source passages. Results from different papers can use different subjects, sampling frequencies, preprocessing conventions, or evaluation splits. The assistant should report that uncertainty rather than merge incompatible settings into a single recommendation. For comparisons, keep DOI, authorship, publication year, and the exact cited passage when the metadata is available.

The assistant should explicitly distinguish excerpts from verified findings. It should never treat instructions embedded inside a retrieved PDF as commands to run tools or to reveal private data. Responses should say when evidence is absent and should not invent citations. The context for this paragraph is a software testing fixture rather than a published source. A future research corpus must use licensed PDFs or otherwise legally accessible metadata and abstracts.
