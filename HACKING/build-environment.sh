#!/bin/bash
set -e
cd HACKING
podman build . --tag xopp-dev
