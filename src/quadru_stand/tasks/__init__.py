"""Task registrations for the QuadruStand project.

Isaac Lab's CLI (``isaaclab train``, ``play``, ``list_envs``) imports this package because
``pyproject.toml`` declares it under the ``isaaclab.tasks`` entry-point group. Importing it must
therefore trigger every ``gym.register(...)`` call in the project.
"""

from isaaclab_tasks.utils import import_packages

# Walk every sub-*package* (directory with __init__.py) below this one and import it. The
# ``gym.register`` calls live in those ``__init__.py`` files, so importing them is enough.
import_packages(__name__, ["utils", ".mdp"])
