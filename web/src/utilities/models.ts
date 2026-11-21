import type {LessonId} from '../../core';
import type {ModelEntry} from '../apis/models';

export function firstCourseModel(catalog:ModelEntry[] | null,course:LessonId) {return catalog?.find(entry=>entry.lesson===course);}
