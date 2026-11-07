"""Export-root pytest resource setup for codedoctor.

Binds the resource-marker hook and the per-user-directory isolation at the
repository root so they reach tests living OUTSIDE the ``codedoctor`` package --
``tests/`` in particular. A conftest hook applies to its own directory and below,
so a package-level conftest alone would leave those tests' resource markers
with no timeout, no CI skip policy, and no error to say so.
"""

from typing import Final

from codedoctor.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


# Honored only in the rootdir conftest: each invocation claims its own basetemp,
# so concurrent sessions never clean up one another's directories.
pytest_plugins: Final = ("codedoctor.lib.testing.basetemp",)

__all__ = ["isolate_user_dirs", "pytest_collection_modifyitems", "pytest_configure"]
