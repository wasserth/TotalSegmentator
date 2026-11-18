"""Focused argument handling for optional NORA mask tagging."""

from pathlib import Path
from unittest.mock import patch

from totalsegmentator.bin import totalseg_combine_masks as cli


def test_nora_tag_literal_arguments(tmp_path):
    """Pass the output path and NORA project identifier literally."""
    output = tmp_path / 'combined mask;literal.nii.gz'
    project = 'project with spaces;literal'
    image = object()
    argv = ['totalseg_combine_masks', '-i', str(tmp_path), '-o', str(output),
            '-m', 'lung', '--nora_tag', project]
    with patch('sys.argv', argv), \
         patch.object(cli, 'combine_masks', return_value=image) as combine, \
         patch.object(cli.nib, 'save') as save, \
         patch.object(cli.subprocess, 'call', return_value=0) as call:
        cli.main()
    combine.assert_called_once_with(Path(tmp_path).absolute(), 'lung', multilabel=False)
    save.assert_called_once_with(image, output.absolute())
    call.assert_called_once_with([
        '/opt/nora/src/node/nora', '-p', project,
        '--add', str(output.absolute()), '--addtag', 'mask',
    ])
