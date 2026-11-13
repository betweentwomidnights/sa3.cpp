"""Promotion boundary tests; no credentials, network or release mutations."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('preview',Path(__file__).parents[1]/'ci/publish-linux-preview.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
SOURCE='a'*40
GGML='b'*40
VERSION='v1.2.3'
PLATFORM='linux-arm64'

def zip_bytes(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for name,data in files.items():archive.writestr(name,data)
    return stream.getvalue()

def artifact(**changes):
    info=dict(service=p.SERVICE,version=VERSION,platform=PLATFORM,commit=SOURCE,ggml_commit=GGML,
              dirty=False,tests_run=True,build_flavor='portable',backends=['vulkan'])
    info.update(changes)
    names=[f'{p.SERVICE}-{VERSION}-{PLATFORM}-{part}.zip' for part in ('core','vulkan')]
    files={names[0]:zip_bytes({'BUILD-INFO.json':json.dumps(info)}),
           names[1]:zip_bytes({'libggml-vulkan.so':b'verified fixture'})}
    files['SHA256SUMS-'+PLATFORM]=''.join(p.sha(files[n])+'  '+n+'\n' for n in names).encode()
    data=zip_bytes(files)
    return dict(digest='sha256:'+p.sha(data),expired=False,workflow_run={'head_sha':SOURCE}),data

class PreviewTests(unittest.TestCase):
    def test_only_exact_successful_repository_source_is_accepted(self):
        run=dict(head_repository={'id':7},head_sha=SOURCE,status='completed',conclusion='success',
                 event='workflow_dispatch',path='.github/workflows/release.yml')
        p.source_run(run,7,SOURCE)
        for key,value in [('head_repository',{'id':8}),('head_sha','c'*40),('conclusion','failure'),
                          ('status','in_progress'),('path','.github/workflows/untrusted.yml')]:
            with self.subTest(key=key),self.assertRaises(ValueError):p.source_run(dict(run,**{key:value}),7,SOURCE)

    def test_outer_digest_and_inner_checksums_are_required(self):
        meta,data=artifact()
        files,info=p.checked_artifact(meta,data,PLATFORM,VERSION,SOURCE,GGML)
        self.assertEqual(info['commit'],SOURCE)
        with self.assertRaises(ValueError):p.checked_artifact(meta,data+b'changed',PLATFORM,VERSION,SOURCE,GGML)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:files={n:archive.read(n) for n in archive.namelist()}
        name=next(n for n in files if n.endswith('-vulkan.zip'))
        files[name]=zip_bytes({'libggml-vulkan.so':b'tampered'})
        changed=zip_bytes(files);meta['digest']='sha256:'+p.sha(changed)
        with self.assertRaises(ValueError):p.checked_artifact(meta,changed,PLATFORM,VERSION,SOURCE,GGML)

    def test_dirty_untested_native_or_different_builds_are_refused(self):
        for change in [dict(dirty=True),dict(tests_run=False),dict(build_flavor='gpu-smoke'),
                       dict(commit='c'*40),dict(ggml_commit='c'*40),dict(backends=['cuda']),
                       dict(version='v1.2.4'),dict(platform='linux-x64')]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                meta,data=artifact(**change);p.checked_artifact(meta,data,PLATFORM,VERSION,SOURCE,GGML)

    def test_local_publish_cannot_use_the_users_token(self):
        with patch.dict(p.os.environ,{},clear=True),self.assertRaises(ValueError):p.publish({},Path('.'))

    def test_existing_asset_is_never_silently_overwritten(self):
        with self.assertRaises(ValueError):p.assert_asset({'size':1,'digest':'sha256:'+'0'*64},
            {'bytes':1,'sha256':'1'*64},'unused')

    def test_drafts_are_found_by_authenticated_listing_when_tag_endpoint_is_404(self):
        draft={'id':42,'tag_name':'linux-v1.2.3-preview.1','draft':True}
        def response(path,optional=False):
            if '/releases/tags/' in path:return None
            if '/releases?' in path:return [draft]
            if path.endswith('/releases/42'):return draft
            raise AssertionError(path)
        with patch.object(p,'api',side_effect=response):
            self.assertEqual(p.release_by_tag('repos/test/repo',draft['tag_name']),draft)

if __name__=='__main__':unittest.main()
