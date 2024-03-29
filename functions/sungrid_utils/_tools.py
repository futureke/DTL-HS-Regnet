import subprocess as sp
import xmltodict


def qstat(qstat_path='qstat', xml_option='-xml'):

    xml = qstat2xml(qstat_path=qstat_path)
    return xml2queue_and_job_info(xml)


def qstat2xml(qstat_path='qstat', xml_option='-xml'):

    try:
        qstatxml = sp.check_output([qstat_path, xml_option], stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print('qstat returncode:', e.returncode)
        print('qstat std output:', e.output)
        raise
    except FileNotFoundError as e:
        e.message = 'Maybe "'+qstat_path+' '+xml_option+'" is not installed.'
        raise
    return qstatxml


def xml2queue_and_job_info(qstatxml):

    x = xmltodict.parse(qstatxml)
    queue_info = []
    if x['job_info']['queue_info'] is not None:
        for job in x['job_info']['queue_info']['job_list']:
            queue_info.append(dict(job))
    job_info = []
    if x['job_info']['job_info'] is not None:
        if isinstance(x['job_info']['job_info']['job_list'], list):
            for job in x['job_info']['job_info']['job_list']:
                job_info.append(dict(job))
        else:
            job_info.append(dict(x['job_info']['job_info']['job_list']))
    return queue_info, job_info