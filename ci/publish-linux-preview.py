#!/usr/bin/env python3
"""Promote exact, verified Linux CI artifacts; never rebuild or overwrite them."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import time
import zipfile

SERVICE = 'sa3'
REPOSITORY = 'betweentwomidnights/sa3.cpp'
PLATFORMS = ('linux-x64', 'linux-arm64')

def gh(*args, optional=False):
    for attempt in range(3):
        result = subprocess.run(['gh', *map(str,args)], capture_output=True, timeout=120)
        if not result.returncode: return result.stdout
        error = result.stderr.decode(errors='replace')
        if optional and 'HTTP 404' in error: return None
        transient=any(word in error for word in ('TLS handshake timeout','timed out','connection reset','HTTP 502','HTTP 503','HTTP 504','unexpected EOF'))
        if args[0]=='api' and transient and attempt<2:
            time.sleep(attempt+1); continue
        error=re.sub(r'(https?://[^?\s\"]+)\?[^\s\"]+',r'\1?[redacted]',error)
        raise RuntimeError(error[-2000:])

def api(path, optional=False):
    raw = gh('api', path, optional=optional)
    return None if raw is None else json.loads(raw)

def sha(data): return hashlib.sha256(data).hexdigest()

def source_run(run, repo_id, source):
    if (run['head_repository']['id'] != repo_id or run['head_sha'] != source
            or run['status'] != 'completed' or run['conclusion'] != 'success'
            or run['event'] not in {'workflow_dispatch','pull_request'}
            or run['path'] not in {'.github/workflows/release.yml','.github/workflows/linux-packaging.yml'}):
        raise ValueError('Refusing an unsuccessful, foreign, or unexpected source workflow run')

def checked_artifact(artifact, data, platform, version, source, ggml):
    if artifact.get('expired') or artifact['digest'] != 'sha256:'+sha(data):
        raise ValueError('GitHub artifact digest/expiry mismatch')
    if artifact['workflow_run']['head_sha'] != source:
        raise ValueError('Artifact source differs from the approved workflow commit')
    names = [f'{SERVICE}-{version}-{platform}-{part}.zip' for part in ('core','vulkan')]
    sums_name = 'SHA256SUMS-'+platform
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        if sorted(outer.namelist()) != sorted([*names,sums_name]):
            raise ValueError('Artifact does not contain exactly core/Vulkan ZIPs and checksums')
        files = {name:outer.read(name) for name in [*names,sums_name]}
    sums = {}
    for line in files[sums_name].decode('utf-8').splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (\S+)',line)
        if not match or match[2] in sums: raise ValueError('Malformed or duplicate checksum entry')
        sums[match[2]] = match[1]
    if set(sums) != set(names) or any(sha(files[n]) != sums[n] for n in names):
        raise ValueError('Inner package checksum mismatch')
    for name in names:
        with zipfile.ZipFile(io.BytesIO(files[name])) as package:
            for entry in package.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
                    raise ValueError('Unsafe package entry')
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Runtime ZIP aliases must be ordinary files')
            if name == names[1] and package.namelist() != ['libggml-vulkan.so']:
                raise ValueError('Unexpected Vulkan overlay contents')
    with zipfile.ZipFile(io.BytesIO(files[names[0]])) as core:
        info = json.loads(core.read('BUILD-INFO.json'))
    if (info.get('service') != SERVICE or info.get('version') != version
            or info.get('platform') != platform or info.get('commit') != source
            or info.get('ggml_commit') != ggml or info.get('dirty') is not False
            or info.get('tests_run') is not True or info.get('build_flavor') != 'portable'
            or info.get('backends') != ['vulkan']):
        raise ValueError('Package provenance does not match the approved portable Vulkan build')
    return files, info

def plan(run_id, source, number, output):
    prefix='repos/'+REPOSITORY
    repo=api(prefix)
    run=api(prefix+f'/actions/runs/{run_id}')
    source_run(run,repo['id'],source)
    jobs=api(prefix+f'/actions/runs/{run_id}/jobs?per_page=100')['jobs']
    for platform in PLATFORMS:
        matched=[j for j in jobs if j['name'].endswith(platform+' Vulkan')]
        if len(matched)!=1 or matched[0]['conclusion']!='success':
            raise ValueError('Both native Linux packaging jobs must pass')
    commit=api(prefix+'/commits/'+source)
    tree=api(prefix+'/git/trees/'+commit['commit']['tree']['sha'])['tree']
    ggml=next(e['sha'] for e in tree if e['path']=='ggml' and e['type']=='commit')
    cmake=api(prefix+'/contents/CMakeLists.txt?ref='+source)
    text=base64.b64decode(cmake['content']).decode()
    match=re.search(r'project\([^)]*?\bVERSION\s+(\d+\.\d+\.\d+)',text)
    if not match: raise ValueError('No source engine version')
    version='v'+match[1]
    tag=f'linux-{version}-preview.{number}'
    artifacts=api(prefix+f'/actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
    all_files={}; provenance={}
    for platform in PLATFORMS:
        expected=f'{SERVICE}-{platform}-vulkan-dry-run'
        matches=[a for a in artifacts if a['name']==expected]
        if len(matches)!=1: raise ValueError('Expected exactly one artifact: '+expected)
        artifact=matches[0]
        if artifact['size_in_bytes']>100*1024*1024: raise ValueError('Unexpected artifact size')
        if artifact['workflow_run']['head_repository_id']!=repo['id']:
            raise ValueError('Artifact came from another repository')
        data=gh('api',prefix+f"/actions/artifacts/{artifact['id']}/zip")
        files,info=checked_artifact(artifact,data,platform,version,source,ggml)
        all_files.update(files)
        provenance[platform]={'artifact_id':artifact['id'],'artifact_digest':artifact['digest'],
            'packages':{n:sha(b) for n,b in files.items()},'build_info':info}
    index={'schema_version':1,'service':SERVICE,'engine_version':version,'release_tag':tag,
        'source_commit':source,'ggml_commit':ggml,'workflow_run':run_id,
        'workflow_url':run['html_url'],'platforms':provenance,
        'policy':'Experimental native Linux Vulkan; no CPU generation, CUDA redistributables or weights bundled.'}
    all_files['LINUX-PREVIEW.json']=(json.dumps(index,indent=2,sort_keys=True)+'\n').encode()
    notes=f'''Experimental Linux Vulkan packages for {SERVICE} {version}, preview {number}.

<!-- gary-linux-preview-v1 {source} -->

Promoted byte-for-byte from the successful native x64/ARM64 [CI run]({run['html_url']}).
Source: `{source}`; pinned ggml: `{ggml}`. See LINUX-PREVIEW.json and each core's
BUILD-INFO.json for exact toolchains and ELF requirements. Extract core plus the
matching Vulkan overlay. Verify SHA256SUMS before use.

Requires a supported Vulkan GPU, system loader and driver. CPU generation is not
a validation target. Model weights and CUDA user libraries are not included.
The GB10-specific CUDA smoke builds are not portable CUDA releases. Hosted x64
checks establish packaging, not x64 GPU inference. This preview does not change
existing Windows/macOS assets or mark a new latest release.
'''
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Use a fresh preview output directory')
    for name,data in all_files.items(): (output/name).write_bytes(data)
    result={'repository':REPOSITORY,'tag':tag,'source_commit':source,'engine_version':version,
        'prerelease':True,'make_latest':False,'release_url':f'https://github.com/{REPOSITORY}/releases/tag/{tag}',
        'assets':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in all_files.items()}}
    (output/'RELEASE-PLAN.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (output/'RELEASE-NOTES.md').write_text(notes)
    return result

def assert_asset(asset,expected,prefix):
    if asset['size']!=expected['bytes']: raise ValueError('Existing asset size mismatch')
    digest=asset.get('digest')
    if digest is None:
        data=gh('api','-H','Accept: application/octet-stream',prefix+f"/releases/assets/{asset['id']}")
        digest='sha256:'+sha(data)
    if digest!='sha256:'+expected['sha256']: raise ValueError('Existing asset digest mismatch')

def publish(result,output):
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('GITHUB_REPOSITORY')!=REPOSITORY:
        raise ValueError('Publish only through the repository workflow and its GITHUB_TOKEN')
    prefix='repos/'+REPOSITORY; tag=result['tag']; source=result['source_commit']
    existing=api(prefix+'/releases/tags/'+tag,optional=True)
    ref=api(prefix+'/git/ref/tags/'+tag,optional=True)
    if ref and (ref['object']['type']!='commit' or ref['object']['sha']!=source):
        raise ValueError('Existing tag targets another source; tags are never moved')
    marker=f'<!-- gary-linux-preview-v1 {source} -->'
    if existing and (marker not in existing['body'] or not existing['prerelease']):
        raise ValueError('Refusing a release not created by this promotion contract')
    if existing and not existing['draft']:
        assets={a['name']:a for a in existing['assets']}
        if set(assets)!=set(result['assets']): raise ValueError('Published asset set differs')
        for n,expected in result['assets'].items(): assert_asset(assets[n],expected,prefix)
        print('Already published and verified:',existing['html_url']); return
    if not existing:
        gh('release','create',tag,'--repo',REPOSITORY,'--target',source,'--draft','--prerelease','--latest=false',
           '--title',f'{SERVICE} {result["engine_version"]} Linux Vulkan preview',
           '--notes-file',output/'RELEASE-NOTES.md')
        existing=api(prefix+'/releases/tags/'+tag)
    assets={a['name']:a for a in existing['assets']}
    if not set(assets).issubset(result['assets']): raise ValueError('Unexpected existing draft assets')
    for name,expected in result['assets'].items():
        if name in assets: assert_asset(assets[name],expected,prefix)
        else: gh('release','upload',tag,output/name,'--repo',REPOSITORY)
    # Verify all uploaded bytes while still draft; no partial release is public.
    existing=api(prefix+'/releases/tags/'+tag)
    assets={a['name']:a for a in existing['assets']}
    if set(assets)!=set(result['assets']): raise ValueError('Draft asset set differs')
    for n,expected in result['assets'].items(): assert_asset(assets[n],expected,prefix)
    gh('release','edit',tag,'--repo',REPOSITORY,'--draft=false','--prerelease','--latest=false')
    released=api(prefix+'/releases/tags/'+tag)
    if released['draft'] or not released['prerelease']: raise ValueError('Unexpected final release state')
    (output/'PUBLISHED.json').write_text(json.dumps({'url':released['html_url'],'tag':tag,'source':source},indent=2)+'\n')
    print('Published verified preview:',released['html_url'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True,type=int)
    p.add_argument('--source-sha',required=True)
    p.add_argument('--preview',type=int,default=1)
    p.add_argument('--output',type=Path,default=Path('dist-linux-preview'))
    p.add_argument('--publish',action='store_true')
    args=p.parse_args()
    if not re.fullmatch('[a-f0-9]{40}',args.source_sha) or args.run_id<1 or args.preview<1:
        p.error('Use a full approved source SHA and positive run/preview numbers')
    result=plan(args.run_id,args.source_sha,args.preview,args.output)
    print(json.dumps(result,indent=2))
    if args.publish: publish(result,args.output)

if __name__=='__main__': main()
