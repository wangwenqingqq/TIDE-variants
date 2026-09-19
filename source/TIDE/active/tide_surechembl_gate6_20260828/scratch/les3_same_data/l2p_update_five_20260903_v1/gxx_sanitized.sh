#!/usr/bin/env bash
exec g++ -fsanitize=address,undefined -fno-omit-frame-pointer "$@"
