"""Explicit task ownership and atomic, inspectable experiment records."""
import contextlib
import csv
import fcntl
import json
import os
from pathlib import Path
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('status\nNO_ROWS\n')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


@contextlib.contextmanager
def lease(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.writer.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'WRITER_ALREADY_ACTIVE: {directory}') from None
        yield


@contextlib.contextmanager
def task(root, task_id, kind, dependencies=()):
    output = Path(root) / 'tasks' / task_id
    with lease(output):
        status_path = output / 'status.json'
        previous = read(status_path) if status_path.exists() else {}
        if previous.get('status') == 'COMPLETE':
            yield None
            return
        if previous.get('status') in ('FAILED', 'DEPENDENCY_FAILED'):
            raise RuntimeError('EXPLICIT_REPAIR_REQUIRED: ' + task_id)
        attempts = sorted(output.glob('attempt_*.json'))
        attempt = len(attempts) + 1
        record = dict(task_id=task_id, kind=kind, status='RUNNING', pid=os.getpid(),
                      physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU'),
                      attempt=attempt, start=time.time(), cwd=str(Path.cwd()),
                      dependencies=list(dependencies))
        write(status_path, record)
        write(output / f'attempt_{attempt:02d}.json', record)
        try:
            yield output
        except BaseException as error:
            record.update(status='FAILED', end=time.time(), error=repr(error),
                          traceback=traceback.format_exc())
            write(output / f'attempt_{attempt:02d}.json', record)
            write(status_path, record)
            raise
        else:
            record.update(status='COMPLETE', end=time.time(), wall_seconds=time.time()-record['start'])
            write(output / f'attempt_{attempt:02d}.json', record)
            write(status_path, record)


def exact_complete(root, ids):
    statuses = {i: read(Path(root)/'tasks'/i/'status.json')
                if (Path(root)/'tasks'/i/'status.json').exists() else {'status':'PENDING'}
                for i in ids}
    if any(s['status'] != 'COMPLETE' for s in statuses.values()):
        raise RuntimeError('MATRIX_NOT_COMPLETE: '+str({i:s['status'] for i,s in statuses.items() if s['status']!='COMPLETE'}))
    assert all(s['task_id'] == i for i,s in statuses.items())
    return statuses
