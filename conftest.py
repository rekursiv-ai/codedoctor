"""Export-root pytest resource setup for codedoctor.

Binds the resource-marker hook and the per-user-directory isolation at the
repository root so they reach tests living OUTSIDE the ``codedoctor`` package --
``tests/`` in particular. A conftest hook applies to its own directory and below,
so a package-level conftest alone would leave those tests' resource markers
with no timeout, no CI skip policy, and no error to say so.
"""

from codedoctor.lib.testing.resources import (
    isolate_user_dirs,
    pytest_collection_modifyitems,
    pytest_configure,
)


__all__ = ["isolate_user_dirs", "pytest_collection_modifyitems", "pytest_configure"]
