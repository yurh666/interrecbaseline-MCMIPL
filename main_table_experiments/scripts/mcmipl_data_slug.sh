#!/usr/bin/env bash
# Map MCMIPL DATA_NAME -> tmp slug
data_slug() {
  case "$1" in
    LAST_FM_STAR) echo last_fm_star ;;
    YELP_STAR) echo yelp_star ;;
    BOOK) echo book ;;
    MOVIE) echo movie ;;
    *) echo "Unknown DATA_NAME: $1" >&2; return 1 ;;
  esac
}
