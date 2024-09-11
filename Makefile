.PHONY: deps_table_update

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src

exclude_folders :=  ""

# Update src/transformers/dependency_versions_table.py
deps_table_update:
	@python setup.py deps_table_update

deps_table_check_updated:
	@md5sum src/transformers/dependency_versions_table.py > md5sum.saved
	@python setup.py deps_table_update
	@md5sum -c --quiet md5sum.saved || (printf "\nError: the version dependency table is outdated.\nPlease run 'make fixup' or 'make style' and commit the changes.\n\n" && exit 1)
	@rm md5sum.saved

# Make marked copies of snippets of codes conform to the original
fix-copies:
#	python utils/check_copies.py --fix_and_overwrite
#	python utils/check_table.py --fix_and_overwrite
	python utils/check_dummies.py --fix_and_overwrite
#	python utils/check_doctest_list.py --fix_and_overwrite
#	python utils/check_docstrings.py --fix_and_overwrite

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/