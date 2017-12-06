import tqdm

problemAreas = dict(
    project=[(1000, 1050)],
    challenge=[(103,163),(162,194)],
    harder_challenge=[
        (160, 235), (265, 355), 
        (380, 405), (581, 620), 
        (652, 705), (731, 777),
        (800, 833), (853, 911),
        (972, 1017), (1061, 1091),
        
    ]
)


class VidList:
    def __init__(self, vids):
        self.vids = vids
    def _repr_html_(self):
        return '<br/>'.join([vid.data for vid in self.vids])


class SpotChecker:

    def __init__(self, allFrames=None):
        if allFrames is None:
            from utils import loadFrames
            allFrames = loadFrames()
        self.allFrames = allFrames

    def __call__(self, laneFinder, videoLabel, whichProblem='all'):
        pipeline = laneFinder.colorFilter
        if whichProblem == 'all':
            problems = problemAreas[videoLabel] 
        else:
            problems = [problemAreas[videoLabel][whichProblem]]
        vids = []
        for problem in tqdm.tqdm_notebook(problems, unit='problem'):
            start, end = problem
            frames = self.allFrames[videoLabel][start:end]
            vid = laneFinder.process(
                frames,
                'tests/%s-%d_%d.mp4' % (videoLabel, start, end), 
                showSteps=True,
                frame0=start,
            )
            frame = frames[int(len(frames)*.8)]
            pipeline(frame)
            pipeline.showMembersFast(
                show=True, 
                title='%s (%d, %d): red/yellow color map is for yellow lines; blue/green for white' % (videoLabel, start, end)
            );
            vids.append(vid)
        vids.append(laneFinder.process(
            frames, 
            'tests/%s-%d_%d-result.mp4' % (videoLabel, start, end),
            showSteps=False,
            frame0=start,
        ))
        return VidList(vids)