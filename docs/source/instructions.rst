Instructions
===== 

.. _instructions:


Installation
------------

EDMTracksLosslessS3Upload is a PowerShell script for uploading local lossless music files to Amazon S3. The script includes:

- Recording outputs using the ``Start-Transcript`` cmdlet.
- Checking there are files in the local folder.
- Checking the files are lossless music.
- Checking if the files are already present in S3.
- Checking if uploads have been successful.
- Moving files to different locations depending on successful or failing upload check.

This document is supported by Uploading Music Files To Amazon S3 (PowerShell Mix) on amazonwebshark.com.

Please use the most recent version. Previous versions are included for completeness.

.. _usage:

Usage
------------

When everything is in place, run the PowerShell script. PowerShell will then move through the script, producing outputs as work is completed. A typical example of a successful transcript is as follows:

.. code-block:: console

              **********************
              Transcript started, output file is C:\Users\Files\EDMTracksLosslessS3Upload.log
              Counting files in local folder.
              2 Local Files Found

              Checking extensions are valid for each local file.
              Acceptable .flac file.
              Acceptable .flac file.
              
              Checking if local files already exist in S3 bucket.
              Checking S3 bucket for MarkOtten-Tranquility-OriginalMix.flac
              MarkOtten-Tranquility-OriginalMix.flac does not currently exist in S3 bucket.
              Checking S3 bucket for StephenJKroos-Micrsh-OriginalMix.flac
              StephenJKroos-Micrsh-OriginalMix.flac does not currently exist in S3 bucket.

              Starting S3 Upload Of 2 Local Files.
              These files are as follows: MarkOtten-Tranquility-OriginalMix StephenJKroos-Micrsh-OriginalMix.flac

              Starting S3 Upload Of MarkOtten-Tranquility-OriginalMix.flac
              Starting S3 Upload Check Of MarkOtten-Tranquility-OriginalMix.flac
              S3 Upload Check Success: MarkOtten-Tranquility-OriginalMix.flac.  Moving to local Success folder
              Starting S3 Upload Of StephenJKroos-Micrsh-OriginalMix.flac
              Starting S3 Upload Check Of StephenJKroos-Micrsh-OriginalMix.flac
              S3 Upload Check Success: StephenJKroos-Micrsh-OriginalMix.flac.  Moving to local Success folder
              All files processed.  Exiting.
              **********************
              Windows PowerShell transcript end
              End time: 20220617153926
              **********************
