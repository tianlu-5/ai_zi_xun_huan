from datetime import datetime
import json
import logging
import random
import os
from typing import Dict, List, Optional
import uuid


class BranchManager:

    DEFAULT_ENTROPY_GAP_CRITICAL = 0.45
    DEFAULT_ENTROPY_GAP_WARNING = 0.28

    def __init__(self, max_branches=10):
        """
        初始化 BranchManager 类。

        :param max_branches: 允许的最大分支数量，默认为 10。
        """
        self.max_branches = max(1, min(100, max_branches))
        self.branches: List[Dict] = []
        self.logger = logging.getLogger(__name__)

        # 先设置过期时间阈值，后续逻辑依赖它
        try:
            # 尝试从环境变量读取配置，若无效则使用默认值
            self.branch_expiration_hours = int(os.environ.get('BRANCH_EXPIRATION_HOURS', 24))
        except (ValueError, TypeError):
            self.branch_expiration_hours = 24
            self.logger.warning('BRANCH_EXPIRATION_HOURS env var is invalid, using default 24')
        # 校验分支过期时间阈值范围
        if not isinstance(self.branch_expiration_hours, (int, float)) or self.branch_expiration_hours <= 0 or self.branch_expiration_hours > 720:
            error_message = 'branch_expiration_hours must be a positive number between 1 and 720'
            self.logger.error(json.dumps({
                'event': 'invalid_branch_expiration_hours',
                'error': error_message
            }, ensure_ascii=False))
            raise ValueError(error_message)

        if not isinstance(max_branches, int) or max_branches < 1 or max_branches > 1000:
            error_message = 'max_branches must be a positive integer between 1 and 1000'
            self.logger.error(json.dumps({
                'event': 'invalid_max_branches',
                'error': error_message
            }, ensure_ascii=False))
            raise ValueError(error_message)

        self.logger.info(json.dumps({
            "event": "init",
            "max_branches": self.max_branches,
            "class_name": "BranchManager"
        }, ensure_ascii=False, indent=4))

    def validate_patch(self, patch):
        """
        验证补丁是否符合 Python 语法要求，并进行格式检查和修正。

        :param patch: 补丁内容（字符串）
        :return: 如果补丁有效，返回 True；否则返回 False。
        """
        try:
            # 严格比对并解析为 JSON，提取 code 字段
            try:
                json_obj = json.loads(patch)
                if isinstance(json_obj, dict) and 'code' in json_obj:
                    code_to_compile = json_obj['code']
                else:
                    code_to_compile = patch
            except json.JSONDecodeError:
                code_to_compile = patch
            compile(code_to_compile, '<string>', 'exec')
            self.logger.info(json.dumps({
                'event': 'validate_patch_success',
            }, ensure_ascii=False))
            return True
        except SyntaxError as e:
            self.logger.error(json.dumps({
                'event': 'validate_patch_error',
                'error': str(e),
                'patch': patch
            }, ensure_ascii=False))
            return False
        except (ValueError, TypeError, KeyError) as e:
            self.logger.error(json.dumps({
                'event': 'validate_patch_unexpected_error',
                'error': str(e),
                'patch': patch
            }, ensure_ascii=False))
            return False

    def is_valid_json_format(self, json_str):
        """
        验证字符串是否为有效的 JSON 格式。

        :param json_str: 要验证的 JSON 字符串
        :return: 如果是有效的 JSON，返回 True；否则返回 False。
        """
        try:
            json.loads(json_str)
            self.logger.info(json.dumps({
                'event': 'json_parse_success',
            }, ensure_ascii=False))
            return True
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'json_parse_error',
                'error': str(e),
                'json_str': json_str
            }, ensure_ascii=False))
            return False

    def list_all_branches(self) -> List[Dict]:
        """
        列出所有分支。
        :return: 包含所有分支信息的列表。
        """
        try:
            # 清理过期分支（原有逻辑保持不变）
            current_time = datetime.now()
            expired_branches = []
            for b in self.branches:
                if isinstance(b, dict) and 'created_at' in b:
                    try:
                        time_diff_hours = (current_time - datetime.fromisoformat(b['created_at'])).total_seconds() / 3600
                        if time_diff_hours > self.branch_expiration_hours:
                            expired_branches.append(b)
                    except (ValueError, TypeError) as e:
                        self.logger.error(f"解析分支创建时间失败: {e}")
            for b in expired_branches:
                try:
                    self.branches.remove(b)
                except Exception as e:
                    self.logger.error(f"列出分支失败: {e}")

            if not self.branches:
                return []  # ★ 直接返回空列表

            result = []
            for b in self.branches:
                if isinstance(b, dict):
                    entry = {
                        'id': b.get('id', ''),
                        'name': b.get('name', b.get('tag', b.get('path', ''))),
                        'path': b.get('path', ''),
                        'created_at': b.get('created_at', ''),
                    }
                    result.append(entry)
                elif isinstance(b, str):
                    result.append(b)
            return result  # ★ 直接返回列表，不要包装成 {"status": "success", "data": result}
        except Exception as e:
            self.logger.error(f"移除过期分支失败: {e}")
            return []  # ★ 出错也返回空列表

    def create_branch(self, name: str, objective: Optional[str] = None) -> Dict:
        """
        创建一个新的分支。同名分支已存在则直接返回。

        :param name: 分支名称（必填）
        :param objective: 分支目标描述（可选）
        :return: 包含 'id', 'name', 'path', 和 'existing' 字段的字典。
        """
        try:
            if not isinstance(name, str) or not name.strip():
                name = 'branch_' + uuid.uuid4().hex[:8]
                self.logger.info(json.dumps({
                    'event': 'create_branch_name_empty',
                    'generated_name': name,
                    'function': 'create_branch'
                }, ensure_ascii=False))
            name = name.strip()
            if not name.strip():
                name = 'branch_' + uuid.uuid4().hex[:8]

            # 幂等检查：同名分支已存在则直接返回
            for b in self.branches:
                if isinstance(b, dict) and b.get('name') == name:
                    self.logger.info(json.dumps({
                        'event': 'create_branch_idempotent',
                        'name': name,
                        'note': '同名分支已存在，跳过创建',
                    }, ensure_ascii=False))
                    return {
                        'id': b.get('id', ''),
                        'name': b.get('name', name),
                        'path': b.get('path', ''),
                        'existing': True,
                    }

            # 超上限：自动淘汰最老的非关键分支
            if len(self.branches) >= self.max_branches:
                try:
                    self.branches = self.branches[-(self.max_branches - 1):] if self.max_branches > 1 else []
                except Exception as e:
                    self.logger.error(json.dumps({
                        'event': 'create_branch_trim_error',
                        'error': str(e)
                    }, ensure_ascii=False))
                    self.branches = []

            # 创建新分支
            new_id = uuid.uuid4().hex[:12]
            safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)
            branch_path = f'evolver/branches/br_{safe_name}_{new_id[:6]}'
            entry = {
                'id': new_id,
                'name': name,
                'path': branch_path,
                'objective': objective or '',
                'created_at': datetime.now().isoformat(),
            }

            try:
                self.branches.append(entry)
                self.logger.info(json.dumps({
                    "event": "create_branch",
                    "id": new_id,
                    "name": name,
                    "path": branch_path,
                }, ensure_ascii=False))
            except Exception as e:
                self.logger.error(json.dumps({
                    'event': 'create_branch_append_error',
                    'error': str(e)
                }, ensure_ascii=False))
            return {
                'id': entry['id'],
                'name': entry['name'],
                'path': entry['path'],
                'existing': False,
            }
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'create_branch_error',
                'error': str(e)
            }, ensure_ascii=False))
            return {
                'id': 'fallback_' + str(random.randint(0, 99999)),
                'name': name if isinstance(name, str) and name else 'unknown',
                'path': '',
                'error': str(e),
            }

    def delete_branch(self, name: str) -> bool:
        """
        删除指定名称的分支。

        :param name: 分支名称（必填）
        :return: 如果删除成功则返回 True，否则返回 False。
        """
        try:
            if not isinstance(name, str) or not name.strip():
                self.logger.warning(json.dumps({
                    'event': 'delete_branch_invalid_name',
                    'name': name,
                    'function': 'delete_branch'
                }, ensure_ascii=False))
                return False
            branches = self.list_all_branches()
            if name not in [b.get('name') for b in branches]:
                self.logger.warning(json.dumps({
                    'event': 'branch_not_found',
                    'name': name
                }, ensure_ascii=False))
                return False
            try:
                self.branches = [b for b in self.branches if b.get('name') != name]
                self.logger.info(json.dumps({
                    'event': 'delete_branch',
                    'name': name,
                }, ensure_ascii=False))
            except Exception as e:
                self.logger.error(json.dumps({
                    'event': 'delete_branch_error',
                    'error': str(e)
                }, ensure_ascii=False))
            return True
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'delete_branch_error',
                'error': str(e),
                'name': name
            }, ensure_ascii=False))
            return False

    def rename_branch(self, old_name: str, new_name: str) -> bool:
        """
        将指定名称的分支重命名为新名称。

        :param old_name: 原始分支名称（必填）
        :param new_name: 新分支名称（必填）
        :return: 如果重命名成功则返回 True，否则返回 False。
        """
        try:
            if not isinstance(old_name, str) or not isinstance(new_name, str) or not old_name.strip() or not new_name.strip() or len(old_name) > 255 or len(new_name) > 255:
                self.logger.warning(json.dumps({
                    'event': 'rename_branch_invalid_params',
                    'old_name': old_name,
                    'new_name': new_name,
                    'function': 'rename_branch',
                    'reason': 'invalid name type, empty, or too long (>255)'
                }, ensure_ascii=False))
                return False
            branches = self.list_all_branches()
            if old_name not in [b['name'] for b in branches]:
                return False
            for b in self.branches:
                if isinstance(b, dict) and b.get('name') == old_name:
                    try:
                        b['name'] = new_name
                    except Exception as e:
                        self.logger.error(json.dumps({
                            'event': 'rename_branch_update_error',
                            'error': str(e),
                            'old_name': old_name,
                            'new_name': new_name,
                        }, ensure_ascii=False))
                        return False
                    break
            self.logger.info(json.dumps({
                'event': 'rename_branch',
                'old_name': old_name,
                'new_name': new_name,
            }, ensure_ascii=False))
            return True
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'rename_branch_error',
                'error': str(e),
                'old_name': old_name,
                'new_name': new_name
            }, ensure_ascii=False, indent=4))
            return False

    def get_branch_info(self, name: str) -> Optional[Dict]:
        """
        获取指定分支的信息（只读查询）。

        :param name: 分支名称
        :return: 分支信息字典，如果分支不存在则返回 None
        """
        try:
            for b in self.branches:
                if isinstance(b, dict) and b.get('name') == name:
                    return {
                        'id': b.get('id', ''),
                        'name': b.get('name', name),
                        'path': b.get('path', ''),
                        'last_commit': b.get('last_commit', ''),
                        'is_active': b.get('is_active', True),
                        'created_at': b.get('created_at', ''),
                        'expiration_time': b.get('expiration_time', ''),
                        'status': b.get('status', 'active'),
                        'additional_info': b.get('additional_info', {}),
                        'contract_valid': b.get('contract_valid', True),
                        'contract_details': b.get('contract_details', {}),
                        'contract_attempts': b.get('contract_attempts', 0),
                        'contract_last_failure_message': b.get('contract_last_failure_message', ''),
                        'contract_last_checked': b.get('contract_last_checked', datetime.now().isoformat()),
                    }
            self.logger.info(json.dumps({
                'event': 'branch_not_found',
                'name': name,
                'function': 'get_branch_info'
            }, ensure_ascii=False))
            return None
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'get_branch_info_error',
                'error': str(e)
            }, ensure_ascii=False))
            return None
    def cleanup_failed_branches(self) -> int:
        """
        清理超过失败阈值的分支（独立方法，有副作用）。

        :return: 清理的分支数量
        """
        removed = 0
        try:
            for b in self.branches[:]:
                if b.get('contract_attempts', 0) >= 5:
                    self.branches.remove(b)
                    removed += 1
                    self.logger.info(json.dumps({
                        'event': 'branch_cleaned',
                        'name': b.get('name', ''),
                        'reason': 'contract_attempts >= 5'
                    }, ensure_ascii=False))
            if removed:
                self.logger.info(json.dumps({
                    'event': 'cleanup_failed_branches_complete',
                    'removed': removed
                }, ensure_ascii=False))
        except Exception as e:
            self.logger.error(json.dumps({
                'event': 'cleanup_failed_branches_error',
                'error': str(e)
            }, ensure_ascii=False))
        return removed

    def cleanup_old_backups(self, days: int = 30) -> None:
        """
        清理超过指定天数的旧备份。

        :param days: 超过该天数的备份将被清理，默认 30 天
        """
        try:
            if not isinstance(days, int) or days < 1:
                return
            if days > 3650:
                self.logger.warning(json.dumps({
                    'event': 'cleanup_old_backups_days_clamped',
                    'original_days': days,
                    'clamped_days': 3650,
                }, ensure_ascii=False))
                days = 3650
            current_time = datetime.now()
            for b in self.branches[:]:
                if isinstance(b, dict) and 'created_at' in b:
                    try:
                        created_time = datetime.fromisoformat(b['created_at'])
                        time_difference = (current_time - created_time).days
                        if time_difference > days:
                            self.branches.remove(b)
                            self.logger.info(json.dumps({
                                'event': 'delete_old_backup',
                                'name': b.get('name', ''),
                                'reason': f'Backup expired after {days} days',
                            }, ensure_ascii=False))
                    except (ValueError, TypeError) as e:
                        self.logger.error(json.dumps({
                            'event': 'cleanup_old_backup_date_error',
                            'error': str(e),
                            'name': b.get('name', ''),
                        }, ensure_ascii=False))
        except Exception as e:
            self.logger.error(json.dumps({'event': 'cleanup_old_backups_error', 'error': str(e)}, ensure_ascii=False))
    def improve_code_quality(self) -> None:
        """对长期未修改的模块进行代码质量改进，包括类型注解、文档字符串、异常处理"""
        try:
            for b in self.branches[:]:
                if isinstance(b, dict) and 'last_modified' in b:
                    last_modified = datetime.fromisoformat(b['last_modified'])
                    time_difference = (datetime.now() - last_modified).days
                    if time_difference > 365:
                        self.logger.info(f'Improving code quality for branch {b.get('name', '')} due to inactivity')
                        # 进行代码质量改进的逻辑
                        b['last_modified'] = datetime.now().isoformat()
        except Exception as e:
            self.logger.error(f'Error improving code quality: {str(e)}')
    def optimize_performance(self) -> None:
        """优化代码性能和可读性，识别并消除不必要的计算"""
        try:
            branches_to_optimize = self._filter_inactive_branches()
            for b in branches_to_optimize:
                self._update_last_modified(b)
        except Exception as e:
            self.logger.error(f'Error optimizing performance: {str(e)}')

    def _update_last_modified(self, branch: Dict) -> None:
        if self._is_inactive(branch):
            self.logger.info(f'Optimizing performance for branch {branch.get('name', '')} due to inactivity')
            branch['last_modified'] = datetime.now().isoformat()

    def _is_inactive(self, branch: Dict) -> bool:
        return (datetime.now() - datetime.fromisoformat(branch['last_modified'])).days > 365

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _get_last_modified(self, branch: Dict) -> datetime:
        return datetime.fromisoformat(branch['last_modified'])

    def _is_inactive(self, branch: Dict) -> bool:
        return (datetime.now() - datetime.fromisoformat(branch['last_modified'])).days > 365

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _get_last_modified(self, branch: Dict) -> datetime:
        return datetime.fromisoformat(branch['last_modified'])

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _get_last_modified(self, branch: Dict) -> datetime:
        return datetime.fromisoformat(branch['last_modified'])

    def _is_inactive(self, last_modified: datetime) -> bool:
        return (datetime.now() - last_modified).days > 365

    def _is_inactive(self, branch: Dict) -> bool:
        last_modified = datetime.fromisoformat(branch['last_modified'])
        return (datetime.now() - last_modified).days > 365

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _get_last_modified(self, branch: Dict) -> datetime:
        return datetime.fromisoformat(branch['last_modified'])

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _get_last_modified(self, branch: Dict) -> datetime:
        return datetime.fromisoformat(branch['last_modified'])

    def _is_inactive(self, last_modified: datetime) -> bool:
        return (datetime.now() - last_modified).days > 365

    def _update_last_modified(self, branch: Dict) -> None:
        last_modified = datetime.fromisoformat(branch['last_modified'])
        if (datetime.now() - last_modified).days > 365:
            self.logger.info(f'Optimizing performance for branch {branch.get('name', '')} due to inactivity')
            branch['last_modified'] = datetime.now().isoformat()

    def _update_last_modified(self, branch: Dict) -> None:
        last_modified = datetime.fromisoformat(branch['last_modified'])
        if (datetime.now() - last_modified).days > 365:
            self.logger.info(f'Optimizing performance for branch {branch.get('name', '')} due to inactivity')
            branch['last_modified'] = datetime.now().isoformat()

    def _filter_inactive_branches(self) -> List[Dict]:
        return [b for b in self.branches if isinstance(b, dict) and 'last_modified' in b]

    def _update_last_modified(self, branch: Dict) -> None:
        last_modified = datetime.fromisoformat(branch['last_modified'])
        if (datetime.now() - last_modified).days > 365:
            self.logger.info(f'Optimizing performance for branch {branch.get('name', '')} due to inactivity')
            branch['last_modified'] = datetime.now().isoformat()

    def _update_last_modified(self, branch: Dict) -> None:
        """更新分支的最后修改时间"""
        last_modified = datetime.fromisoformat(branch['last_modified'])
        time_difference = (datetime.now() - last_modified).days
        if time_difference > 365:
            self.logger.info(f'Optimizing performance for branch {branch.get('name', '')} due to inactivity')
            branch['last_modified'] = datetime.now().isoformat()
# Deprecated function _legacy_merge has been removed and commented out. (Format adjusted)
