"""Labelled reference transcriptions of remote modelling code (read as text, never executed).

Each module transcribes one family's forward pass as plain torch (plus stock transformers / library
code where the remote file builds on it). Its header names the source repo, the revision, the sha256
of each file transcribed and the line ranges, so the reference can be re-checked against the remote
code by reading it.
"""
