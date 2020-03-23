#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version

run_tests() {
    py.test -v --cov=./
}

if [[ "$RUN_LINT" == "true" ]]; then
    echo "Running linter and mypy"
    pylint --disable-warnings kglm
    mypy --ignore-missing-imports --no-strict-optional kglm
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
